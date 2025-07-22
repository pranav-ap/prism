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
        return f"[{topics_str}] : {self.text[:50]}{'...' if len(self.text) > 50 else ''}"


def get_sample_tweets():
    with open("D:/code/prism/data/sample_tweets.json", "r") as f:
        tweets = json.load(f)
        tweets = [Tweet(**tweet) for tweet in tweets]

    return tweets
