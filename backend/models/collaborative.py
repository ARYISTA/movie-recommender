"""
models/collaborative.py — Collaborative Filtering with SVD

HOW IT WORKS (simple explanation):
────────────────────────────────────────────────────────────
Imagine a giant grid (matrix) where:
  - Each ROW    = one user
  - Each COLUMN = one movie
  - Each CELL   = the rating that user gave that movie (or blank)

Most cells are blank (users only rate a few movies).
SVD (Singular Value Decomposition) fills in the blanks by finding
hidden patterns — e.g. "users who liked Inception also liked Interstellar."

It decomposes the matrix into:
  R ≈ U × Σ × Vᵀ
where:
  U  = user feature matrix  (what each user likes)
  Σ  = importance of each feature
  Vᵀ = movie feature matrix (what each movie is about)

The dot product U[user] · Vᵀ[movie] predicts the rating.

We use the Surprise library which provides an optimised SVD
that achieves ~0.87 RMSE on MovieLens — near state-of-the-art.
────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Surprise is a dedicated collaborative filtering library
try:
    from surprise import SVD, Dataset, Reader, accuracy
    from surprise.model_selection import cross_validate, train_test_split
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    # Avoid printing at import time (and avoid non-ASCII output on Windows consoles).
    # The rest of the app can run without collaborative filtering installed.


class CollaborativeRecommender:
    """
    SVD-based collaborative filtering.
    Falls back gracefully if Surprise is not installed.
    """

    def __init__(self, artifacts_dir: str | Path = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.trainset = None
        self.all_movie_ids: list[int] = []

    # ── Training ────────────────────────────────────────────────────────────

    def fit(self, ratings_df: pd.DataFrame, n_epochs: int = 20) -> dict:
        """
        Train SVD on ratings_df with columns: [userId, movieId, rating].
        Returns cross-validation metrics.
        """
        if not SURPRISE_AVAILABLE:
            print("Surprise not available. Skipping collaborative training.")
            return {}

        print("Training SVD collaborative filter...")

        # Store all movie IDs for later use (making predictions for all movies)
        self.all_movie_ids = ratings_df["movieId"].unique().tolist()

        # Surprise needs a specific data format
        # Reader tells it the rating scale
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            ratings_df[["userId", "movieId", "rating"]],
            reader
        )

        # SVD hyperparameters (tuned for MovieLens)
        self.model = SVD(
            n_factors=100,    # number of latent features (hidden dimensions)
            n_epochs=n_epochs,
            lr_all=0.005,     # learning rate
            reg_all=0.02,     # regularisation to prevent overfitting
            biased=True,      # include user + item bias terms
        )

        # 5-fold cross-validation to measure RMSE and MAE
        print("Running 5-fold cross-validation (this takes ~2 min on MovieLens 25M)...")
        cv_results = cross_validate(self.model, data, measures=["RMSE", "MAE"], cv=5)

        avg_rmse = np.mean(cv_results["test_rmse"])
        avg_mae  = np.mean(cv_results["test_mae"])
        print(f"SVD CV - RMSE: {avg_rmse:.4f} | MAE: {avg_mae:.4f}")

        # Retrain on the full dataset for production use
        full_trainset = data.build_full_trainset()
        self.model.fit(full_trainset)
        self.trainset = full_trainset

        return {"rmse": avg_rmse, "mae": avg_mae}

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self) -> None:
        if self.model is None:
            return
        joblib.dump(self.model,         self.artifacts_dir / "svd_model.pkl")
        joblib.dump(self.all_movie_ids, self.artifacts_dir / "cf_movie_ids.pkl")
        print("Collaborative model saved.")

    def load(self) -> bool:
        try:
            self.model         = joblib.load(self.artifacts_dir / "svd_model.pkl")
            self.all_movie_ids = joblib.load(self.artifacts_dir / "cf_movie_ids.pkl")
            print("Collaborative model loaded from disk.")
            return True
        except FileNotFoundError:
            return False

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict how a user would rate a movie (0.5 – 5.0 scale)."""
        if self.model is None:
            return 0.0
        prediction = self.model.predict(str(user_id), str(movie_id))
        return prediction.est  # estimated rating

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_ids: list[int] | None = None,
    ) -> list[dict]:
        """
        Predict ratings for all movies, return top_n highest.
        Returns [{"movie_id": int, "score": float}, ...]
        where score is the predicted rating normalised to [0, 1].
        """
        if self.model is None:
            return []

        exclude_set = set(exclude_ids or [])
        predictions = []

        for movie_id in self.all_movie_ids:
            if movie_id in exclude_set:
                continue
            pred = self.model.predict(str(user_id), str(movie_id))
            predictions.append({
                "movie_id": int(pred.iid),
                "score":    (pred.est - 0.5) / 4.5,  # normalise to 0–1
            })

        # Sort by predicted score, return top N
        predictions.sort(key=lambda x: x["score"], reverse=True)
        return predictions[:top_n]
