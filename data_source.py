import json
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Tweet:
    id: str
    text: str
    timestamp: str
    # label → score
    topics: Optional[Dict[str, float]] = None

    def __repr__(self):
        top_topics = sorted(self.topics.items(), key=lambda x: -x[1])
        topics_str = ', '.join(f"{t}:{s:.2f}" for t, s in top_topics if s >= 0.5)
        return f"[{topics_str}] : {self.text[:30]}{'...' if len(self.text) > 50 else ''}"


@dataclass
class TweetPair:
    tweet1: Tweet
    tweet2: Tweet
    similarity_score: float
    contradiction_score: Optional[float] = None

    def __repr__(self):
        return (
            f"Similarity Score : {self.similarity_score}, Contradiction Score : {self.contradiction_score}\n"
            f"- {self.tweet1.text[:50]}{'...' if len(self.tweet1.text) > 50 else ''}\n"
            f"- {self.tweet2.text[:50]}{'...' if len(self.tweet2.text) > 50 else ''}\n"
        )


def get_sample_tweets():
    with open("D:/code/prism/data/sample_tweets.json", "r") as f:
        tweets = json.load(f)
        tweets = [Tweet(**tweet) for tweet in tweets]

    return tweets
