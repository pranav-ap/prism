from typing import List
from transformers import pipeline
from similarity import TweetPair
from more_itertools import chunked


class ContradictionDetector:
    def __init__(self, threshold: float = 0.5, batch_size: int = 8):
        self.threshold = threshold
        self.batch_size = batch_size
        self.model = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli"
        )

    def _get_scores(self, premises: List[str], hypotheses: List[str]) -> List[float]:
        inputs = [f"{p} </s> {h}" for p, h in zip(premises, hypotheses)]
        results = self.model(inputs, top_k=3)

        return [
            r["score"]
            for result in results
            for r in result
            if r["label"].lower() == "contradiction"
        ]

    def detect(self, pairs: List[TweetPair]) -> List[TweetPair]:
        for batch in chunked(pairs, self.batch_size):
            premises = [pair.tweet1.text for pair in batch]
            hypotheses = [pair.tweet2.text for pair in batch]
            scores = self._get_scores(premises, hypotheses)

            for pair, score in zip(batch, scores):
                pair.contradiction_score = score

        return [
            pair for pair in pairs
            if pair.contradiction_score >= self.threshold
        ]
