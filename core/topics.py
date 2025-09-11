from typing import Dict, List
from bertopic import BERTopic
from transformers import pipeline
from bertopic.representation import TextGeneration

from core.data_source import Tweet
from prefect import task


class TopicClassifier:
    def __init__(self):
        prompt = "I have a topic described by the following keywords: [KEYWORDS]. Based on the previous keywords, what is this topic about?"
        generator = pipeline('text2text-generation', model='google/flan-t5-base')

        representation_model = TextGeneration(
            generator,
            prompt=prompt,
        )

        self.topic_model = BERTopic(
            representation_model=representation_model
        )

    @task
    def classify(self, tweets: List[Tweet]):
        tweet_texts = [tweet.text for tweet in tweets]

        # Learn topics from data & Assign each tweet a topic and a probability
        topics, probs = self.topic_model.fit_transform(tweet_texts)

        # gives each topic a human-readable name instead of just an ID (0, 1, 2...)
        labels = self.topic_model.generate_topic_labels(
            nr_words=3,
            topic_prefix=False,
            word_length=15,
            separator=", "
        )

        self.topic_model.set_topic_labels(labels)

        available_topics = self.topic_model.get_topics().keys()
        label_mapping: Dict[str, str] = dict(zip(
            available_topics,
            labels,
        ))

        THRESHOLD = 0.5

        for tweet, topic, prob in zip(tweets, topics, probs):
            if prob < THRESHOLD:
                continue

            if topic == -1:
                tweet.topics = {"other": 1.0}
            else:
                topic = str(topic)
                tweet.topics = {label_mapping[topic]: prob}

        return tweets

