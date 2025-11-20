import numpy as np
import time
from functools import lru_cache

# ------------------------------------------------------------
# DATASET GENERATION
# ------------------------------------------------------------
def generate_dataset(num_users, num_items, sparsity):
    """
    Generates a user-item rating matrix with given sparsity.
    """
    matrix = np.random.rand(num_users, num_items)
    mask = np.random.rand(num_users, num_items) < sparsity
    matrix[mask] = 0
    return matrix


# ------------------------------------------------------------
# OPTIMIZED COSINE SIMILARITY
# ------------------------------------------------------------
def cosine_similarity_opt(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return np.dot(a, b) / denom


# ------------------------------------------------------------
# RECOMMENDATION ENGINE
# ------------------------------------------------------------
def get_top_k_recommendations(matrix, user_id, k):
    num_users = matrix.shape[0]
    target = matrix[user_id]

    sims = []
    for other in range(num_users):
        if other == user_id:
            continue
        sim = cosine_similarity_opt(target, matrix[other])
        sims.append((other, sim))

    return sorted(sims, key=lambda x: x[1], reverse=True)[:k]


# ------------------------------------------------------------
# STRESS TEST
# ------------------------------------------------------------
def stress_test(size):
    print("\nRunning stress test…")
    matrix = generate_dataset(size, size // 2, sparsity=0.92)

    t0 = time.time()
    get_top_k_recommendations(matrix, user_id=0, k=5)
    t1 = time.time()

    return t1 - t0


# ------------------------------------------------------------
# MENU
# ------------------------------------------------------------
def main():
    print("\n===== RECOMMENDATION SYSTEM (Phase 3) =====\n")

    # User input for dataset
    num_users = int(input("Enter number of users (e.g., 500): "))
    num_items = int(input("Enter number of items (e.g., 300): "))
    sparsity = float(input("Enter sparsity level (0.0 to 1.0, typical = 0.92): "))

    # Generate dataset
    print("\nGenerating dataset…")
    matrix = generate_dataset(num_users, num_items, sparsity)
    print("Dataset generated successfully!")

    # Recommendation input
    user_id = int(input("\nEnter user ID to get recommendations: "))
    k = int(input("Enter number of recommendations (Top-K): "))

    print(f"\nTop-{k} recommendations for user {user_id}:")
    results = get_top_k_recommendations(matrix, user_id, k)

    for r in results:
        print(f"User {r[0]} → Similarity: {round(r[1], 4)}")

    # Optional stress test
    run_stress = input("\nRun stress test? (y/n): ").lower()
    if run_stress == "y":
        size = int(input("Enter stress test dataset size (e.g., 1500): "))
        t = stress_test(size)
        print(f"Stress test completed in {round(t, 4)} seconds.")

    print("\n===== Phase 3 run completed =====\n")


if __name__ == "__main__":
    main()