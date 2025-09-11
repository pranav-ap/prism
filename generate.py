import json
from dataclasses import dataclass, asdict
from typing import List, Literal, TypeAlias
import litellm


LabelType: TypeAlias = Literal["Contradiction", "No Contradiction", "Not Comparable"]


@dataclass
class SyntheticTweetPair:
    tweet1: str
    tweet2: str
    label: str
    reason: str


class SyntheticTweetDatasetGenerator:
    def __init__(self):
        self.model = "ollama/llama3.2:3b-instruct-fp16"
        self.api_base = "http://localhost:11434"

    @staticmethod
    def _build_prompt(label: LabelType) -> str:
        return f"""Generate a pair of tweets with the following relationship: "{label}".

Requirements:
- Return your answer as a JSON object with the following keys: "tweet1", "tweet2", "label", "reason".
- The tweets should sound realistic and casual.
- Keep each tweet under 30 words.

Example format:
{{
  "tweet1": "Tweet content...",
  "tweet2": "Tweet content...",
  "label": "{label}",
  "reason": "Short explanation."
}}"""

    def _generate_one(self, label: LabelType) -> SyntheticTweetPair:
        message = self._build_prompt(label)
        response = litellm.completion(
            model=self.model,
            api_base=self.api_base,
            temperature=0.5,
            messages=[{"role": "user", "content": message}],
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

        data = json.loads(response["choices"][0]["message"]["content"].strip())

        return SyntheticTweetPair(**data)

    def generate(self, labels: List[LabelType], n_per_class: int) -> List[SyntheticTweetPair]:
        return [self._generate_one(label) for label in labels for _ in range(n_per_class)]


def main():
    generator = SyntheticTweetDatasetGenerator()
    labels: List[LabelType] = ["Contradiction", "No Contradiction", "Not Comparable"]
    dataset = generator.generate(labels, n_per_class=2)

    with open('output/generated_tweets.json', 'w', encoding='utf-8') as f:
        json.dump([asdict(pair) for pair in dataset], f, ensure_ascii=True, indent=2)


if __name__ == '__main__':
    main()
