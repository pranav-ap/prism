from typing import List, Set, Tuple
from itertools import combinations

import faiss
from sentence_transformers import SentenceTransformer
from prefect import task

from core.data_source import Tweet, TweetPair


class SimilarityFinder:
    def __init__(self, threshold: float, k: int):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        self.threshold = threshold
        self.k = k

    def _get_embeddings(self, tweets: List[Tweet]):
        texts = [tweet.text for tweet in tweets]
        return self.encoder.encode(texts, convert_to_numpy=True)

    @staticmethod
    def _build_faiss_index(embeddings):
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def find_similar_pairs(self, tweets: List[Tweet]) -> List[TweetPair]:
        if len(tweets) < 2:
            return []

        embeddings = self._get_embeddings(tweets)
        index = self._build_faiss_index(embeddings)

        scores, indices = index.search(embeddings, k=self.k)
        seen: Set[Tuple[int, int]] = set()
        similar_pairs: List[TweetPair] = []

        for i, (score_row, idx_row) in enumerate(zip(scores, indices)):
            # skip self-match. just loop over others
            for j, score in zip(idx_row[1:], score_row[1:]):
                if score >= self.threshold:
                    pair = tuple(sorted((i, j)))

                    if pair not in seen:
                        seen.add(pair)
                        similar_pairs.append(TweetPair(
                            tweet1=tweets[pair[0]],
                            tweet2=tweets[pair[1]],
                            similarity_score=score
                        ))

        return similar_pairs

    def find_similar_pairs_between(self, tweets_a: List[Tweet], tweets_b: List[Tweet]) -> List[TweetPair]:
        if not tweets_a or not tweets_b:
            return []

        emb_a = self._get_embeddings(tweets_a)
        emb_b = self._get_embeddings(tweets_b)

        index = self._build_faiss_index(emb_b)
        # for each in A, get top-k in B
        scores, indices = index.search(emb_a, k=self.k)

        seen: Set[Tuple[str, str]] = set()
        similar_pairs: List[TweetPair] = []

        for i, (score_row, idx_row) in enumerate(zip(scores, indices)):
            for j, score in zip(idx_row, score_row):
                if score >= self.threshold:
                    t1 = tweets_a[i]
                    t2 = tweets_b[j]

                    key = tuple(sorted((t1.id, t2.id)))
                    if key in seen:
                        continue

                    seen.add(key)

                    similar_pairs.append(TweetPair(
                        tweet1=t1,
                        tweet2=t2,
                        similarity_score=score
                    ))

        return similar_pairs

    @task
    def detect(self, tweets_by_user):
        all_pairs = []

        for u1, u2 in combinations(tweets_by_user.keys(), 2):
            print(f'Finding similar pairs between {u1} and {u2}')

            similar_pairs = self.find_similar_pairs_between(
                tweets_by_user[u1],
                tweets_by_user[u2]
            )

            print(f'Found {len(similar_pairs)} similar pairs')

            all_pairs.extend(similar_pairs)

        return all_pairs
