from typing import List

import faiss
from sentence_transformers import SentenceTransformer

from data_source import Tweet


def get_tweet_embeddings(tweets: List[Tweet]):
    texts = [tweet.text for tweet in tweets]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, convert_to_numpy=True)


def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def find_similar_pairs(tweets: List[Tweet]):
    similar_pairs = []
    THRESHOLD = 0.3

    embeddings = get_tweet_embeddings(tweets)
    index = build_faiss_index(embeddings)

    scores, indices = index.search(embeddings, k=5)
    seen = set()

    for i, (score_row, idx_row) in enumerate(zip(scores, indices)):
        # skip self (idx 0). Just loop over others
        for j, score in zip(idx_row[1:], score_row[1:]):
            if score >= THRESHOLD:
                # sort to avoid (A,B) vs (B,A)
                pair = tuple(sorted((i, j)))
                if pair not in seen:
                    seen.add(pair)
                    similar_pairs.append((tweets[i], tweets[j], score))

    return similar_pairs
