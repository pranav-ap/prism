import json
from dataclasses import dataclass, asdict
from typing import List

import litellm


@dataclass
class SyntheticTweetPair:
    tweet1: str
    tweet2: str
    label: str
    reason: str


class SyntheticTweetDatasetGenerator:
    def __init__(self):
        pass

    def _build_prompt(self, label: str) -> str:
        message = f"""Generate a pair of tweets with the following relationship: "{label}".

Requirements:
- Return your answer as a JSON object with the following keys: "tweet1", "tweet2", "label", "reason".
- The tweets should sound realistic and casual.
- The label should match one of the three relationships - ["Contradiction", "No Contradiction", "Not Comparable"].
- Keep each tweet under 30 words.

Example format:
{{
  "tweet1": "Tweet content...",
  "tweet2": "Tweet content...",
  "label": "{label}",
  "reason": "Short explanation."
}}
"""

        return message

    def _generate_one(self, label: str) -> SyntheticTweetPair:
        message = self._build_prompt(label)

        response = litellm.completion(
            model="ollama/llama3.2:3b-instruct-fp16",
            messages=[{
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
                            "tweet1": {"type": "string"},
                            "tweet2": {"type": "string"},
                            "label": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["tweet1", "tweet2", "label", "reason"]
                    }
                }
            },
        )

        response = response["choices"][0]["message"]["content"]
        response = response.strip().lower()
        response = json.loads(response)

        return SyntheticTweetPair(**response)

    def generate(self, labels: List[str], n_per_class: int) -> List[SyntheticTweetPair]:
        all_pairs = []

        for label in labels:
            for _ in range(n_per_class):
                pair = self._generate_one(label)
                all_pairs.append(pair)

        return all_pairs


def main():
    generator = SyntheticTweetDatasetGenerator()

    labels = ["Contradiction", "No Contradiction", "Not Comparable"]
    dataset = generator.generate(labels, n_per_class=2)

    with open('output/generated_tweets.json', 'w', encoding='utf-8') as f:
        dataset = [asdict(pair) for pair in dataset]
        json.dump(dataset, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
