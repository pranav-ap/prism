from transformers import pipeline


def classify_topics(tweets, candidate_labels):
    classifier = pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    tweet_texts = [tweet.text for tweet in tweets]

    results = classifier(tweet_texts, candidate_labels, multi_label=True)

    THRESHOLD = 0.5

    for tweet, res in zip(tweets, results):
        filtered = {
            label: score
            for label, score in zip(res["labels"], res["scores"])
            if score >= THRESHOLD
        }

        tweet.topics = filtered if filtered else {"other": 1.0}
