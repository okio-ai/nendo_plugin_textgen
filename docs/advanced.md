# Advanced Usage

When calling the plugin, we provide a default system prompt.
You can also provide your own system prompt as follows:

```python
from nendo import Nendo

nd = Nendo(plugins=["nendo_plugin_textgen"])
nd.plugins.textgen(prompt=["Tell me about your favorite song."], system_prompt=["You're a pirate."])
```

## Batched Calling

Per default the plugin takes in and returns a batch of prompts and answers. 
So you can call the plugin with a list of prompts for additional speedup:

```python
from nendo import Nendo

nd = Nendo(plugins=["nendo_plugin_textgen"])
two_answers = nd.plugins.textgen(
    prompt=[
        "Tell me about your favorite song.", 
        "Tell me about your favorite instrument."
    ], 
    system_prompt=[
        "You're a pirate."
        "You're a five year old child."
    ]
)
```

## Prompt presets

We provide a few templates for common use cases. You can use them to get started quickly.

### Summarization

```
You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. 
        Think step-by-step and create a concise overview of the key findings, their strategic significance, and the actionable insights. 
        Extract the core ideas, key themes, essential facts, primary arguments, pivotal events, and significant outcomes. 
        Then, distill this information into a concise summary of no more than 5000 words, organized by order of importance. 
        Ensure that the summary is coherent, captures the main points accurately, and omits any extraneous details. 
        Aim for clarity and succinctness. The summary should be tailored for a general audience and maintain a neutral tone. 
        The summary will be used for an executive briefing, so focus on clarity and precision. 
        Ensure that excerpts contain all relevant context needed to interpret them - in other words don't extract small snippets that are missing important context.
        REPLY ONLY WITH THE SUMMARY TEXT:
        Here's an example:
        "The quick brown fox jumps over the lazy dog."
```

```python
from nendo import Nendo

nd = Nendo(plugins=["nendo_plugin_textgen"])
my_text = "...."
summary = nd.plugins.textgen.summarization(my_text)
```

### Topic Detection

```
You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. 
        Think step-by-step and identify and list 3 to 20 keywords that succinctly encapsulate the main topics, sentiments, and entities present. 
        Arrange these keywords in order of their relevance to the text's central message. 
        For each keyword, categorize it under 'Concepts', 'Events', 'People', 'Organisations', 'Places', 'Dates', whichever is most fitting.
        REPLY ONLY WITH A LIST OF THE KEYWORDS AND THEIR CATEGORIES AND NOTHING ELSE.
        Here's an example:
        "Concepts: AI, ML, NLP\nEvents: World War II, The Great Depression\nPeople: Albert Einstein, Marie Curie\nOrganisations: Google, Microsoft\nPlaces: New York, Paris\nDates: 2022, 2023"
```

```python
from nendo import Nendo

nd = Nendo(plugins=["nendo_plugin_textgen"])
my_text = "...."
topics = nd.plugins.textgen.topic_detection(my_text)
```

### Sentiment Analysis

```
You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. 
        Think step-by-step and conduct a detailed sentiment analysis by identifying the primary sentiment (positive, negative, or neutral) and any secondary sentiments present. 
        Rate the intensity of the primary sentiment on a scale from 1 (very negative) to 5 (very positive).
        REPLY ONLY WITH THE SENTIMENTS.
        Here's an example:
        "Primary Sentiment: Positive\nSecondary Sentiments: Neutral, Neutral, Neutral\nIntensity: 5"
```

```python
from nendo import Nendo

nd = Nendo(plugins=["nendo_plugin_textgen"])
my_text = "...."
sentiments = nd.plugins.textgen.sentiment_analysis(my_text)
```

### MusicGen enhancement
```
You write descriptive prompts for a text to music ai. Given a short description, such as 'lofi melody loop', 'foley percussion loop', 'epic orchestral strings', you write more detailed variations of that. include additional information when relevant, such as genre, key, bpm, instruments, overall style or atmosphere, and words like 'high quality', 'crisp mix', 'virtuoso', 'professionally mastered', or other enhancements.
        here's one possible way a prompt could be formatted:
        'lofi melody loop, A minor, 110 bpm, jazzy chords evoking a feeling of curiosity, relaxing, vinyl recording'

        Write 5 prompts for the given topic in a similar style. be descriptive! only describe the relevant elements - we don't want drums in a melody loop, nor melody or bass in a percussion loop. 
        we also don't need to describe atmosphere for a drum loop. note: the text to music model cannot make vocals, so don't write prompts for with them. 
        Also, for melody loops, you can specify 'no drums' in your prompt. 
        I'd like them to be varied and high quality. Please write a prompt for the following description:
```

### Audio Metadata Summary 

```
Please use the following key and values of a description of an audio file to generate a summary of its contents.
        Make sure to keep the summary brief, but include all the information
        especially things like tempo, key, scale and other audio features:"
```





