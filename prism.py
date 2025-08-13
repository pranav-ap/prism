from core.contra import ContradictionDetectorLiteLLM
from core.data_source import get_extracted_tweets
from core.report import generate_report
from core.similarity import SimilarityFinder
from core.topics import TopicClassifier
from core.translator import Translator


def prepare_tweets(usernames, CANDIDATE_LABELS):
    translator = Translator()
    topic_classifier = TopicClassifier(candidate_labels=CANDIDATE_LABELS)

    tweets_by_user = {}

    for index, username in enumerate(usernames):
        print(f'Processing tweets for user: {username}')
        user_tweets = get_extracted_tweets(username)

        if True:
            translator.lazy_translate(user_tweets)

        topic_classifier.classify(user_tweets)
        tweets_by_user[username] = user_tweets

    if len(usernames) == 1:
        username = usernames[0]
        print(f'Only one user ({username}) provided. Self-comparison will be performed.')
        tweets_by_user[f'{username}_2'] = tweets_by_user[username]

    return tweets_by_user


def main():
    usernames = [
        'krystalball',
        # 'nytimes',
        # 'nytimeses'
    ]

    CANDIDATE_LABELS = [
        "politics",
        "entertainment",
        "sports",
        "science",
        "technology",
    ]

    tweets_by_user = prepare_tweets(
        usernames,
        CANDIDATE_LABELS
    )

    similarity_finder = SimilarityFinder(threshold=0.5, k=5)
    similar_pairs = similarity_finder.detect(tweets_by_user)
    print(f"Detected {len(similar_pairs)} unique similar pairs")
    del similarity_finder

    contradictions_detector = ContradictionDetectorLiteLLM(threshold=0.5)
    contradictions = contradictions_detector.detect(similar_pairs)
    print(f"Found {len(contradictions)} total contradictions!")
    del contradictions_detector

    generate_report(contradictions)
    print("Done!")


if __name__ == '__main__':
    main()
