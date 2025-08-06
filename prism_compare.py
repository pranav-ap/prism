from rich.progress import track

from core.data_source import get_extracted_tweets
from core.similarity import TweetSimilarityFinder
from core.snoopy import ContradictionDetectorLiteLLM
from core.topics import classify_topics
from core.translator import Translator
from core.report import generate_report


def main():
    usernames = ['nytimes', 'nytimeses']
    tweets_by_user = {}
    candidate_labels = ["politics", "entertainment", "sports", "science", "technology"]
    translator = Translator()

    for index, username in enumerate(usernames):
        print(f'Processing tweets for user: {username}')

        user_tweets = get_extracted_tweets(username)

        for t in user_tweets:
            t.original_text = t.text

            if index != 0:
                t.text = translator.translate(t.original_text)

        classify_topics(user_tweets, candidate_labels)
        tweets_by_user[username] = user_tweets

    similarity_finder = TweetSimilarityFinder(threshold=0.5, k=5)
    contradictions_detector = ContradictionDetectorLiteLLM(threshold=0.5)

    candidate_labels += ["other"]
    all_pairs = []

    for topic in track(candidate_labels, total=len(candidate_labels), description="Processing topics"):
        print(f'=> Processing topic: {topic}')

        subsets = {
            user: [t for t in tweets_by_user[user] if t.topics and topic in t.topics]
            for user in usernames
        }

        similar_pairs = similarity_finder.find_similar_pairs_between(subsets[usernames[0]], subsets[usernames[1]])
        print(f'Found {len(similar_pairs)} similar pairs')

        all_pairs.extend(similar_pairs)

    print(f"Total unique similar pairs collected: {len(all_pairs)}")

    contradictions = contradictions_detector.detect(all_pairs)
    print(f"Found {len(contradictions)} total contradictions")

    generate_report(contradictions)
    print("Done!")


if __name__ == '__main__':
    main()
