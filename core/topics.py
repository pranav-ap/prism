from transformers import pipeline


class TopicClassfier:
    def __init__(self, candidate_labels):
        self.candidate_labels = candidate_labels

        self.classifier = pipeline(
            task="zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    def classify(self, tweets):
        tweet_texts = [tweet.text for tweet in tweets]

        results = self.classifier(
            tweet_texts,
            self.candidate_labels,
            multi_label=True
        )

        THRESHOLD = 0.5

        for tweet, res in zip(tweets, results):
            filtered = {
                label: score
                for label, score in zip(res["labels"], res["scores"])
                if score >= THRESHOLD
            }

            tweet.topics = filtered if filtered else {"other": 1.0}
