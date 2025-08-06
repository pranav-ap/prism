import json
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ScraperTweet:
    id: int
    text: str
    timestamp: str

    def __repr__(self):
        return f"{self.text[:60]}{'...' if len(self.text) > 60 else ''}"

    def __eq__(self, other):
        return isinstance(other, ScraperTweet) and self.text == other.text

    def __hash__(self):
        return hash(self.text)


@dataclass
class Tweet:
    id: int
    text: str
    timestamp: str
    # label → score
    topics: Optional[Dict[str, float]] = None

    def __repr__(self):
        topics_str = "-"

        if self.topics:
            top_topics = sorted(self.topics.items(), key=lambda x: -x[1])
            topics_str = ', '.join(f"{t}:{s:.2f}" for t, s in top_topics)

        return f"[{topics_str}] : {self.text[:60]}{'...' if len(self.text) > 60 else ''}"


@dataclass
class TweetPair:
    tweet1: Tweet
    tweet2: Tweet

    similarity_score: float
    contradiction_score: Optional[float] = None
    contradiction_reason: Optional[str] = None

    def __repr__(self):
        return (
            f"Similarity Score : {self.similarity_score}, Contradiction Score : {self.contradiction_score}\n"
            f"Reason : {self.contradiction_reason}\n"
            f"- {self.tweet1.text[:50]}{'...' if len(self.tweet1.text) > 50 else ''}\n"
            f"- {self.tweet2.text[:50]}{'...' if len(self.tweet2.text) > 50 else ''}\n"
        )


def get_sample_tweets():
    with open("./data/sample_tweets.json", "r") as f:
        tweets = json.load(f)
        tweets = [Tweet(**tweet) for tweet in tweets]

    return tweets


def get_extracted_tweets(username: str):
    with open(f"./output/extracted_tweets_{username}.json", "r") as f:
        tweets = json.load(f)
        tweets = [Tweet(**tweet) for tweet in tweets]

    return tweets


