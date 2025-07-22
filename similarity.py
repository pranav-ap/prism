from typing import List, Set, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from data_source import Tweet, TweetPair


class TweetSimilarityFinder:
    def __init__(self, threshold: float, k: int):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.threshold = threshold
        self.k = k

    def _get_embeddings(self, tweets: List[Tweet]):
        texts = [tweet.text for tweet in tweets]
        return self.model.encode(texts, convert_to_numpy=True)

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
