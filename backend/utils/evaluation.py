"""
utils/evaluation.py — Recommendation System Evaluation Metrics

KEY METRICS EXPLAINED:
──────────────────────────────────────────────────────────
• Precision@K: Of the K movies we recommended, what fraction
  did the user actually like? (liked = rated ≥ 4.0)
  → Measures: "Are our recommendations relevant?"

• Recall@K: Of all the movies the user would have liked,
  what fraction did we successfully recommend in the top K?
  → Measures: "Are we missing many good movies?"

• NDCG@K (Normalised Discounted Cumulative Gain):
  Like Precision but penalises relevant movies ranked lower.
  A relevant movie at rank 1 is worth more than at rank 10.
  → Measures: "Is the ranking order good?"

• RMSE (Root Mean Squared Error): Only for collaborative filtering.
  How far off is our predicted rating from the actual rating?
  RMSE of 0.87 means predictions are ±0.87 stars on average.
  → Measures: "How accurate are our rating predictions?"

Target: Precision@10 ≥ 0.85 (i.e. ≥8.5 of 10 recommendations are relevant)
──────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from typing import Callable


# ── Offline evaluation (using held-out ratings) ───────────────────────────────

def precision_at_k(
    recommended_ids: list[int],
    relevant_ids: set[int],
    k: int = 10,
) -> float:
    """
    Precision@K = (# relevant in top K) / K

    Args:
        recommended_ids: Ordered list of recommended movie IDs
        relevant_ids:    Set of movie IDs the user actually liked
        k:               How many recommendations to consider
    """
    top_k = recommended_ids[:k]
    hits  = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / k


def recall_at_k(
    recommended_ids: list[int],
    relevant_ids: set[int],
    k: int = 10,
) -> float:
    """
    Recall@K = (# relevant in top K) / (total # relevant)
    Returns 0 if there are no relevant items.
    """
    if not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    hits  = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(
    recommended_ids: list[int],
    relevant_ids: set[int],
    k: int = 10,
) -> float:
    """
    NDCG@K — rewards relevant items appearing earlier in the list.
    """
    top_k = recommended_ids[:k]
    dcg   = sum(
        1 / np.log2(rank + 2)      # +2 because rank is 0-indexed
        for rank, mid in enumerate(top_k)
        if mid in relevant_ids
    )
    # Ideal DCG: all relevant items ranked first
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def rmse(predictions: list[float], actuals: list[float]) -> float:
    """Root Mean Squared Error between predicted and actual ratings."""
    preds = np.array(predictions)
    acts  = np.array(actuals)
    return float(np.sqrt(np.mean((preds - acts) ** 2)))


# ── Full evaluation pipeline ──────────────────────────────────────────────────

def evaluate_recommender(
    recommend_fn: Callable[[int, list[int]], list[dict]],
    ratings_df: pd.DataFrame,
    test_size: float = 0.2,
    k: int = 10,
    min_threshold: float = 4.0,     # rating ≥ this = "liked"
    n_users: int = 200,             # sample this many users for speed
) -> dict:
    """
    Leave-one-out evaluation:
      For each test user, hide their most recent highly-rated movie,
      generate recommendations, and check if the hidden movie appears.

    Args:
        recommend_fn:  Callable (user_id, watched_ids) → list of {"movie_id": ...}
        ratings_df:    Full ratings DataFrame
        k:             Evaluate precision/recall at this K
        min_threshold: Rating ≥ this counts as "liked"
        n_users:       How many users to evaluate (sampling for speed)

    Returns:
        dict with mean Precision@K, Recall@K, NDCG@K
    """
    # Sample users who have enough ratings to split
    user_counts = ratings_df.groupby("userId")["rating"].count()
    eligible    = user_counts[user_counts >= 20].index.tolist()

    rng = np.random.default_rng(42)
    sampled_users = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)

    metrics_list = []

    for user_id in sampled_users:
        user_ratings = ratings_df[ratings_df["userId"] == user_id].sort_values("timestamp")

        # Split: keep last 20% as test, rest as train
        split = int(len(user_ratings) * (1 - test_size))
        train_ratings = user_ratings.iloc[:split]
        test_ratings  = user_ratings.iloc[split:]

        # Watched IDs (from training set)
        watched_ids = train_ratings["id"].tolist()

        # Relevant IDs = movies rated highly in test set
        relevant_ids = set(
            test_ratings[test_ratings["rating"] >= min_threshold]["id"].tolist()
        )

        if not relevant_ids:
            continue

        # Get recommendations
        try:
            recs = recommend_fn(user_id, watched_ids)
            rec_ids = [r["id"] for r in recs]
        except Exception:
            continue

        metrics_list.append({
            "precision": precision_at_k(rec_ids, relevant_ids, k),
            "recall":    recall_at_k(rec_ids, relevant_ids, k),
            "ndcg":      ndcg_at_k(rec_ids, relevant_ids, k),
        })

    if not metrics_list:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}

    df = pd.DataFrame(metrics_list)
    return {
        f"precision@{k}": round(df["precision"].mean(), 4),
        f"recall@{k}":    round(df["recall"].mean(),    4),
        f"ndcg@{k}":      round(df["ndcg"].mean(),      4),
        "n_users_evaluated": len(metrics_list),
    }
