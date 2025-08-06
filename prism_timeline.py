from rich.progress import track

from data_source import get_extracted_tweets
from report import generate_report
from similarity import TweetSimilarityFinder
from snoopy import ContradictionDetectorLiteLLM
from topics import classify_topics


def main():
    # username = 'krystalball'
    username = 'nytimes'
    tweets = get_extracted_tweets(username)

    candidate_labels = ["politics", "entertainment", "sports", "science", "technology"]
    classify_topics(tweets, candidate_labels)

    similarity_finder = TweetSimilarityFinder(threshold=0.5, k=5)
    contradictions_detector = ContradictionDetectorLiteLLM(threshold=0.5)

    candidate_labels += ["other"]
    all_pairs = []

    for topic in track(candidate_labels, total=len(candidate_labels), description="Processing topics"):
        print(f'=> Processing tweets for topic: {topic}')
        subset = [t for t in tweets if t.topics and topic in t.topics]
        print(f'Found {len(subset)} tweets')

        similar_pairs = similarity_finder.find_similar_pairs(subset)
        print(f'Found {len(similar_pairs)} similar pairs')

        all_pairs.extend(similar_pairs)

    print(f"Total unique similar pairs collected: {len(all_pairs)}")

    contradictions = contradictions_detector.detect(all_pairs)
    print(f"Found {len(contradictions)} total contradictions")

    generate_report(contradictions)
    print("Done!")


if __name__ == '__main__':
    main()
