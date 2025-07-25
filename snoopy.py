from typing import List
from transformers import pipeline
from similarity import TweetPair
from more_itertools import chunked
import litellm
import json


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


class ContradictionDetectorLiteLLM:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @staticmethod
    def _build_prompt(tweet1: str, tweet2: str) -> str:
        message = (
            f"""You are given two tweets. Determine the logical relationship between them by choosing one of the following answers:
            
- "Contradiction" → if the tweets make opposing or incompatible claims about the same topic or event.
- "No Contradiction" → if the tweets are about the same topic or event, but do not contradict each other.
- "Not Comparable" → if the tweets are about different topics, unrelated events, or lack comparable claims.

Then, provide a short explanation for your answer in the reason field.

Tweet 1: {tweet1}
Tweet 2: {tweet2}

Answer:
"""
        )

        return message

    def _detect(self, tweet1: str, tweet2: str):
        message = self._build_prompt(tweet1, tweet2)

        response = litellm.completion(
            model="ollama/llama3.2:3b-instruct-fp16",
            messages = [{
                "content": message,
                "role": "user"
            }],
            temperature=0,
            api_base="http://localhost:11434",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "reason": {"type": "string"},
                        }
                    }
                }
            },
        )

        response = response["choices"][0]["message"]["content"]
        response = response.strip().lower()
        response = json.loads(response)

        return response

    def detect(self, pairs: List[TweetPair]) -> List[TweetPair]:
        for p in pairs:
            response = self._detect(p.tweet1.text, p.tweet2.text)

            if response["answer"] == "contradiction":
                p.contradiction_score = 1.0
            elif response["answer"] == "no contradiction":
                p.contradiction_score = 0.0
            else:
                p.contradiction_score = None

        return [
            pair for pair in pairs
            if pair.contradiction_score >= self.threshold
        ]