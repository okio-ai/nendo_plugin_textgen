from typing import Any, List

import torch
from nendo import Nendo, NendoUtilityPlugin, NendoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import TextgenConfig, TextgenPromptTemplates
import gc


settings = TextgenConfig()


class Textgen(NendoUtilityPlugin):
    """A nendo plugin for text generation based on transformers by HuggingFace.

    You can use this plugin to generate text from a prompt. You can also use the templates to generate text in a predefined way.

    Examples:
        ```python
        from nendo import Nendo, NendoConfig

        nd = Nendo(config=NendoConfig(plugins=["nendo_plugin_textgen"]))

        text = nd.plugins.textgen(prompt="What is your favorite song?")
        print(text)
        ```
    """

    nendo_instance: Nendo = None
    config: NendoConfig = None
    tokenizer: Any = None
    model: Any = None
    model_loaded: bool = None
    device: str = None

    def __init__(self, **data: Any):
        """Initialize plugin."""
        super().__init__(**data)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False

    def _load_model_and_tokenizer(self):
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.textgen_model,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            use_safetensors=settings.use_safetensors,
            attn_implementation=settings.attn_implementation,
            torch_dtype=torch_dtype,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.textgen_model, trust_remote_code=True, use_fast=True
        )
        # settings to get batch generation to work
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def templates(self) -> TextgenPromptTemplates:
        """Get the prompt templates for the textgen plugin.

        These prompt templates can be used to use the textgen plugin in some predefined ways with other nendo plugins.

        Returns:
            TextgenPromptTemplates: The prompt templates.

        """
        return TextgenPromptTemplates(
            MUSICGEN_PROMPT_ENHANCE=settings.musicgen_enhance_prompt,
            AUDIO_METADATA=settings.audio_metadata_summary_prompt,
            TOPIC_DETECTION=settings.topic_detection_prompt,
            SENTIMENT_ANALYSIS=settings.sentiment_analysis_prompt,
            SUMMARIZATION=settings.summarization_prompt,
        )

    def summarization(
        self,
        prompt: str,
        max_new_tokens: int = settings.max_new_tokens,
    ) -> str:
        """Generate a summary from a prompt.

        Args:
            prompt (str): The prompt to generate the summary from.
            max_new_tokens (int): The maximum number of tokens to generate.

        Returns:
            str: The generated summary.
        """
        return self._generate(
            prompts=[prompt],
            system_prompts=[self.templates().SUMMARIZATION],
            max_new_tokens=max_new_tokens,
        )[0]

    def sentiment_analysis(
        self,
        prompt: str,
        max_new_tokens: int = settings.max_new_tokens,
    ) -> str:
        """Generate a sentiment analysis from a prompt.

        Args:
            prompt (str): The prompt to generate the sentiment analysis from.
            max_new_tokens (int): The maximum number of tokens to generate.

        Returns:
            str: The generated sentiment analysis.
        """
        return self._generate(
            prompts=[prompt],
            system_prompts=[self.templates().SENTIMENT_ANALYSIS],
            max_new_tokens=max_new_tokens,
        )[0]

    def topic_detection(
        self,
        prompt: str,
        max_new_tokens: int = settings.max_new_tokens,
    ) -> str:
        """Generate a topic detection from a prompt.

        Args:
            prompt (str): The prompt to generate the topic detection from.
            max_new_tokens (int): The maximum number of tokens to generate.

        Returns:
            str: The generated topic detection.
        """
        return self._generate(
            prompts=[prompt],
            system_prompts=[self.templates().TOPIC_DETECTION],
            max_new_tokens=max_new_tokens,
        )[0]

    def _generate(
        self,
        prompts: List[str],
        system_prompts: List[str],
        max_new_tokens: int,
    ) -> List[str]:
        if not self.model_loaded:
            self._load_model_and_tokenizer()
            self.model_loaded = True

        with torch.no_grad():
            prompt_templates = [
                f"""<|im_start|>system
                {system_prompt}<|im_end|>
                <|im_start|>user
                {prompt}<|im_end|>
                <|im_start|>assistant
            """
                for prompt, system_prompt in zip(prompts, system_prompts)
            ]

            gen_input = self.tokenizer(
                prompt_templates, return_tensors="pt", padding=True
            ).to(self.device)

            generated_ids = self.model.generate(**gen_input, max_new_tokens=max_new_tokens)

            # Identify the end of the input sequence
            input_ids_end = gen_input.input_ids.size(1)

            # Slice the generated_ids to get only the generated response
            generated_ids = generated_ids[:, input_ids_end:]
            results = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # manually clear memory
            gc.collect()
            if self.device == "cuda:0":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            return results

    @NendoUtilityPlugin.run_utility
    def generate_text(
        self,
        prompts: List[str],
        system_prompts: List[str] = None,
        max_new_tokens: int = settings.max_new_tokens,
    ) -> List[str]:
        """Generate text from a prompt.

        Args:
            prompts (List[str]): The prompts to generate the text from.
            system_prompts (List[str], optional): The system prompts to use. Defaults to None.
            max_new_tokens (int): The maximum number of tokens to generate.

        Returns:
            str: The generated text.
        """
        assert len(prompts) > 0, "Please provide at least one prompt."

        if system_prompts is None:
            system_prompts = [
                settings.default_system_prompt for _ in enumerate(prompts)
            ]

        assert len(prompts) == len(
            system_prompts
        ), "Please provide the same number of prompts and system prompts."

        return self._generate(
            prompts=prompts,
            system_prompts=system_prompts,
            max_new_tokens=max_new_tokens,
        )
