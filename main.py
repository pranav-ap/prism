import pprint
import logfire

from data_source import get_sample_tweets
from similarity import find_similar_pairs
from topics import classify_topics


def setup():
    logfire.configure(token='pylf_v1_eu_7MbBJfzdplBRdSxsQLMg33dxLmnKZkk0TXNvyH2kr44C')


def main():
    # setup()

    tweets = get_sample_tweets()
    classify_topics(tweets)

    candidate_labels = ["politics", "entertainment", "sports", "science", "technology"]

    for topic in candidate_labels:
        print(f'Processing tweets for topic: {topic}')

        subset = [
            tweet
            for tweet in tweets
            if tweet.topics and topic in tweet.topics
        ]

        print(f'Found {len(subset)} tweets for topic "{topic}"')

        if len(subset) < 2:
            continue

        similar_pairs = find_similar_pairs(subset)
        pprint.pprint(similar_pairs)

        # snoop around to find contradictions

    # report the contradictions


if __name__ == '__main__':
    main()
