import unittest
import torch
import random
import numpy as np

from nendo import Nendo, NendoConfig

nd = Nendo(
    config=NendoConfig(
        log_level="INFO",
        plugins=["nendo_plugin_textgen"],
    ),
)

# fix all seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class TextgenTests(unittest.TestCase):
    def test_run_textgen(self):
        nd.library.reset(force=True)
        result = nd.plugins.textgen(prompts=["What is your favorite song?"])
        self.assertIsNotNone(result[0])

    def test_run_textgen_topic_detection(self):
        nd.library.reset(force=True)
        result = nd.plugins.textgen.topic_detection(
            prompt="What is your favorite song?"
        )
        self.assertIsNotNone(result)

    def test_run_textgen_summarization(self):
        nd.library.reset(force=True)
        result = nd.plugins.textgen.summarization(prompt="What is your favorite song?")
        self.assertIsNotNone(result)

    def test_run_textgen_sentiment_analysis(self):
        nd.library.reset(force=True)
        result = nd.plugins.textgen.sentiment_analysis(
            prompt="What is your favorite song?"
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
