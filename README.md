# Nendo Plugin TextGen

<br>
<p align="left">
    <img src="https://okio.ai/docs/assets/nendo_core_logo.png" width="350" alt="nendo core">
</p>
<br>

<p align="left">
<a href="https://okio.ai" target="_blank">
    <img src="https://img.shields.io/website/https/okio.ai" alt="Website">
</a>
<a href="https://twitter.com/okio_ai" target="_blank">
    <img src="https://img.shields.io/twitter/url/https/twitter.com/okio_ai.svg?style=social&label=Follow%20%40okio_ai" alt="Twitter">
</a>
<a href="https://discord.gg/gaZMZKzScj" target="_blank">
    <img src="https://dcbadge.vercel.app/api/server/XpkUsjwXTp?compact=true&style=flat" alt="Discord">
</a>
</p>

---

A text generation plugin using local LLMs or other text generation methods. 
Builds on top of `transformers` by Hugging Face.

## Features 
- Generate text from a prompt using a local LLM
- Access all LLMs from the [Hugging Face Model Hub](https://huggingface.co/models)

## Requirements

Since we depend on `transformers`, please make sure that you fulfill their requirements.
You also need Pytorch installed on your system, please refer to the [pytorch installation instructions](https://pytorch.org/get-started/locally/).

## Installation

1. [Install Nendo](https://github.com/okio-ai/nendo#installation)
2. `pip install nendo-plugin-textgen`

If you have a CUDA GPU you can install the following library for an additional speedup: 

`pip install flash-attn --no-build-isolation`

Then set `ATTN_IMPLEMENTATION=flash_attention_2` in your environment variables.


## Usage

Take a look at a basic usage example below.
For more detailed information, please refer to the [documentation](https://okio.ai/docs/plugins).

```pycon
>>> from nendo import Nendo
>>> nd = Nendo(plugins=["nendo_plugin_textgen"])

>>> nd.plugins.textgen(prompt=["Tell me about your favorite song."])[0]
```


## Contributing

Visit our docs to learn all about how to contribute to Nendo: [Contributing](https://okio.ai/docs/contributing/)


## License

Nendo: MIT License
