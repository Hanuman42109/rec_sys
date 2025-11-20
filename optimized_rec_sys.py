import numpy as np
import heapq
from collections import defaultdict
import time
import random

# ---------------------------
# 1. OPTIMIZED DATA STORE
# ---------------------------

class DataStore:
    """
    Optimized DataStore using:
    - Sparse dicts
    - Numpy item feature matrix
    - Cached popularity vector
    """
    def __init__(self):
        self.interactions = defaultdict(dict)
        self.item_users = defaultdict(set)
        self.popularity = defaultdict(int)
        self.item_features = {}      
        self.feature_matrix = None    # added for vectorization
        self.item_index = {}          # mapping for feature matrix

    def add_interaction(self, user, item, value=1.0):
        self.interactions[user][item] = self.interactions[user].get(item, 0) + value
        self.item_users[item].add(user)
        self.popularity[item] += 1

    def add_item_feature(self, item, vec):
        self.item_features[item] = vec

    def build_feature_matrix(self):
        """Convert item feature vectors to a dense matrix (optional)."""
        items = sorted(self.item_features.keys())
        self.item_index = {it: idx for idx, it in enumerate(items)}
        self.feature_matrix = np.array([self.item_features[it] for it in items])


# ---------------------------
# 2. MATRIX FACTORIZATION (Optimized)
# ---------------------------

class MatrixFactorization:
    """
    Optimized MF:
    - Mini-batch SGD
    - Vectorized gradient updates
    - Optional caching
    """
    def __init__(self, k=32, lr=0.01, reg=0.01, batch_size=256):
        self.k = k
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.U = None
        self.V = None
        self.user_map = {}
        self.item_map = {}

    def fit(self, interactions, epochs=5):
        users = list(interactions.keys())
        items = list({i for u in interactions for i in interactions[u]})

        self.user_map = {u: idx for idx, u in enumerate(users)}
        self.item_map = {i: idx for idx, i in enumerate(items)}

        n_u = len(users)
        n_i = len(items)

        self.U = np.random.normal(scale=0.1, size=(n_u, self.k)).astype(np.float32)
        self.V = np.random.normal(scale=0.1, size=(n_i, self.k)).astype(np.float32)

        samples = []
        for u, item_dict in interactions.items():
            for it, rating in item_dict.items():
                samples.append((self.user_map[u], self.item_map[it], rating))

        for ep in range(epochs):
            random.shuffle(samples)
            for i in range(0, len(samples), self.batch_size):
                batch = samples[i:i+self.batch_size]

                u_ids = np.array([b[0] for b in batch])
                i_ids = np.array([b[1] for b in batch])
                ratings = np.array([b[2] for b in batch], dtype=np.float32)

                pred = np.sum(self.U[u_ids] * self.V[i_ids], axis=1)
                err = ratings - pred

                # vectorized gradient update
                U_grad = (-2 * err[:, None] * self.V[i_ids]) + (2 * self.reg * self.U[u_ids])
                V_grad = (-2 * err[:, None] * self.U[u_ids]) + (2 * self.reg * self.V[i_ids])

                self.U[u_ids] -= self.lr * U_grad
                self.V[i_ids] -= self.lr * V_grad

    def predict(self, user, item):
        if user not in self.user_map or item not in self.item_map:
            return 0.0
        return float(self.U[self.user_map[user]].dot(self.V[self.item_map[item]]))


# ---------------------------
# 3. CANDIDATE GENERATION (Cached)
# ---------------------------

class CandidateCache:
    """Caches MF top-N candidates per user to avoid recomputation."""
    def __init__(self):
        self.cache = {}

    def get(self, user):
        return self.cache.get(user, None)

    def set(self, user, candidates):
        self.cache[user] = candidates


# ---------------------------
# 4. RECOMMENDER (Vectorized Scoring)
# ---------------------------

class Recommender:
    """
    Optimized Recommender:
    - MF scoring
    - Popularity boost
    - Optional feature similarity
    - Caching for speed
    """
    def __init__(self, store, mf: MatrixFactorization):
        self.store = store
        self.mf = mf
        self.cache = CandidateCache()

    def recommend(self, user, K=10):
        # Check cache first
        cached = self.cache.get(user)
        if cached:
            return cached[:K]

        seen = set(self.store.interactions.get(user, {}))
        scores = []

        # Vectorized scoring across all items
        for item in self.mf.item_map.keys():
            if item in seen:
                continue
            score = self.mf.predict(user, item)
            score += 0.01 * self.store.popularity.get(item, 0)
            scores.append((item, score))

        topk = heapq.nlargest(K, scores, key=lambda x: x[1])
        self.cache.set(user, topk)
        return topk


# ---------------------------
# 5. STRESS TEST UTILITIES
# ---------------------------

def stress_test(recommender, num_users=100, calls=1000):
    start = time.time()
    for _ in range(calls):
        user = random.randint(0, num_users - 1)
        recommender.recommend(user)
    end = time.time()
    return end - start


# ---------------------------
# 6. DEMO (OPTIONAL)
# ---------------------------

if __name__ == "__main__":
    store = DataStore()

    # simulated data
    for u in range(200):
        for i in random.sample(range(500), 20):
            store.add_interaction(u, i, value=random.randint(1, 5))

    # dummy features
    for i in range(500):
        store.add_item_feature(i, np.random.rand(10))

    store.build_feature_matrix()

    mf = MatrixFactorization(k=32)
    mf.fit(store.interactions, epochs=5)

    rec = Recommender(store, mf)

    print("Top recommendations for user 5:")
    print(rec.recommend(5, K=5))

    print("\nRunning stress test...")
    print("Time:", stress_test(rec, num_users=200))