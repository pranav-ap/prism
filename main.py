from rich.progress import track

from data_source import get_extracted_tweets
from report import generate_report
from similarity import TweetSimilarityFinder
from snoopy import ContradictionDetector
from topics import classify_topics


def main():
    tweets = get_extracted_tweets()
    classify_topics(tweets)

    contradictions_detector = ContradictionDetector(threshold=0.7)
    similarity_finder = TweetSimilarityFinder(threshold=0.6, k=5)

    candidate_labels = ["politics", "entertainment", "sports", "science", "technology"]

    all_pairs = []
    seen = set()

    for topic in track(candidate_labels, total=len(candidate_labels), description="Processing topics"):
        print(f'=> Processing tweets for topic: {topic}')
        subset = [t for t in tweets if t.topics and topic in t.topics]
        print(f'Found {len(subset)} tweets')

        similar_pairs = similarity_finder.find_similar_pairs(subset)
        print(f'Found {len(similar_pairs)} similar pairs')

        for pair in similar_pairs:
            key = tuple(sorted((pair.tweet1.id, pair.tweet2.id)))
            if key not in seen:
                seen.add(key)
                all_pairs.append(pair)

    print(f"Total unique similar pairs collected: {len(all_pairs)}")

    contradictions = contradictions_detector.detect(all_pairs)
    print(f"Found {len(contradictions)} total contradictions")

    generate_report(contradictions)
    print("Done!")


if __name__ == '__main__':
    main()
