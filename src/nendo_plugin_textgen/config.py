"""Configuration for the textgen plugin."""
import dataclasses

from nendo import NendoConfig
from pydantic import Field


class TextgenConfig(NendoConfig):
    """Configuration for the textgen plugin."""

    textgen_model: str = Field("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    use_safetensors: bool = Field(True)
    attn_implementation: str = Field("eager")
    max_new_tokens: int = Field(100)
    use_bf16: bool = Field(True)
    default_system_prompt: str = Field(
        """The following is a conversation with an AI assistant. 
        The assistant is helpful, creative, clever, and very friendly.
        """,
    )
    topic_detection_prompt: str = Field(
        """You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. 
        Think step-by-step and identify and list 3 to 20 keywords that succinctly encapsulate the main topics, sentiments, and entities present. 
        Arrange these keywords in order of their relevance to the text's central message. 
        For each keyword, categorize it under 'Concepts', 'Events', 'People', 'Organisations', 'Places', 'Dates', whichever is most fitting.
        REPLY ONLY WITH A LIST OF THE KEYWORDS AND THEIR CATEGORIES AND NOTHING ELSE!!
        Here's an example:
        "Concepts: AI, ML, NLP\nEvents: World War II, The Great Depression\nPeople: Albert Einstein, Marie Curie\nOrganisations: Google, Microsoft\nPlaces: New York, Paris\nDates: 2022, 2023"
        """,
    )
    sentiment_analysis_prompt: str = Field(
        """You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. 
        Think step-by-step and conduct a detailed sentiment analysis by identifying the primary sentiment (positive, negative, or neutral) and any secondary sentiments present. 
        Rate the intensity of the primary sentiment on a scale from 1 (very negative) to 5 (very positive).
        REPLY ONLY WITH THE SENTIMENTS!!
        Here's an example:
        "Primary Sentiment: Positive\nSecondary Sentiments: Neutral, Neutral, Neutral\nIntensity: 5"
        """,
    )
    summarization_prompt: str = Field(
        """You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. 
        You create summaries of text transcripts.
        Ensure that the summary is coherent, captures the main points accurately, and omits any extraneous details. 
        Aim for clarity and succinctness. The summary should be tailored for a general audience and maintain a neutral tone. 
        The summary will be used for an executive briefing, so focus on clarity and precision. 
        """,
    )
    audio_metadata_summary_prompt: str = Field(
        """Please use the following key and values of a description of an audio file to generate a summary of its contents.
        Make sure to keep the summary brief, but include all the information
        especially things like tempo, key, scale and other audio features:"
        """,
    )
    musicgen_enhance_prompt: str = Field(
        """You write descriptive prompts for a text to music ai. Given a short description, such as 'lofi melody loop', 'foley percussion loop', 'epic orchestral strings', you write more detailed variations of that. include additional information when relevant, such as genre, key, bpm, instruments, overall style or atmosphere, and words like 'high quality', 'crisp mix', 'virtuoso', 'professionally mastered', or other enhancements.
        here's one possible way a prompt could be formatted:
        'lofi melody loop, A minor, 110 bpm, jazzy chords evoking a feeling of curiosity, relaxing, vinyl recording'

        Write 5 prompts for the given topic in a similar style. be descriptive! only describe the relevant elements - we don't want drums in a melody loop, nor melody or bass in a percussion loop. 
        we also don't need to describe atmosphere for a drum loop. note: the text to music model cannot make vocals, so don't write prompts for with them. 
        Also, for melody loops, you can specify 'no drums' in your prompt. 
        I'd like them to be varied and high quality. Please write a prompt for the following description:
        """,
    )


@dataclasses.dataclass
class TextgenPromptTemplates:
    """Prompt templates for the textgen plugin."""

    MUSICGEN_PROMPT_ENHANCE: str
    AUDIO_METADATA: str
    TOPIC_DETECTION: str
    SENTIMENT_ANALYSIS: str
    SUMMARIZATION: str
