"""Setup file for the nendo-plugin-textgen package."""
from distutils.core import setup

if __name__ == "__main__":
    setup(
        name="nendo-plugin-textgen",
        version="0.1.0",
        description="A text generation plugin using local LLMs or other text generation methods. Builds on top of `transformers` by Hugging Face.",
        author="Aaron Abebe <aaron@okio.ai>",
    )
