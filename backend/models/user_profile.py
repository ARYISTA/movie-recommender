"""
models/user_profile.py — Dynamic User Profile Builder

HOW IT WORKS:
────────────────────────────────────────────────────────────
A user profile is a numerical representation of what a user likes.
We maintain TWO complementary representations:

1. GENRE VECTOR (simple, interpretable)
   A 20-element array, one slot per genre.
   Each time a user watches a movie, we add the genre weights.
   The vector is normalised so we always have a probability distribution.

   e.g. after watching 3 action movies and 1 comedy:
   Action=0.75, Comedy=0.25, Drama=0.0, ...

2. TF-IDF PROFILE VECTOR (powerful, drives content-based recs)
   We average the TF-IDF vectors of all watched movies.
   This captures fine-grained taste: not just "likes Action"
   but "likes action movies directed by Christopher Nolan
   with themes of time and identity."

Both are stored in memory (and periodically synced to DB).
────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from backend.utils.preprocessing import ALL_GENRES, genre_vector
from backend.models.content_based import ContentBasedRecommender


class UserProfileBuilder:
    """
    Builds and updates user preference profiles from watch history.
    """

    def __init__(self, content_model: ContentBasedRecommender, movies_df: pd.DataFrame):
        self.content_model = content_model
        self.movies = movies_df.set_index("id")

    # ── Profile construction ──────────────────────────────────────────────────

    def build_genre_vector(self, watched_movie_ids: list[int]) -> np.ndarray:
        """
        Return a normalised genre preference vector.
        Recent watches count more (exponential recency weighting).
        """
        n = len(watched_movie_ids)
        if n == 0:
            return np.zeros(len(ALL_GENRES), dtype=np.float32)

        # Exponential decay: most recent watch has weight 1.0, oldest near 0
        weights = np.exp(np.linspace(-1, 0, n))
        weights /= weights.sum()

        profile = np.zeros(len(ALL_GENRES), dtype=np.float32)

        for weight, movie_id in zip(weights, watched_movie_ids):
            if movie_id not in self.movies.index:
                continue
            genres_str = str(self.movies.loc[movie_id, "genres"])
            g_list = genres_str.split("|") if genres_str and genres_str != "nan" else []
            gvec = genre_vector(g_list)
            profile += weight * gvec

        # Normalise to sum = 1
        total = profile.sum()
        if total > 0:
            profile /= total

        return profile

    def build_tfidf_vector(self, watched_movie_ids: list[int]) -> np.ndarray | None:
        """
        Average the TF-IDF vectors of all watched movies.
        Returns None if no vectors are available.
        """
        if not watched_movie_ids:
            return None

        vectors = []
        for movie_id in watched_movie_ids:
            vec = self.content_model.get_movie_vector(movie_id)
            if vec is not None:
                # Convert sparse → dense for averaging
                if hasattr(vec, "toarray"):
                    vec = vec.toarray().flatten()
                vectors.append(vec)

        if not vectors:
            return None

        # Simple average (could also do recency-weighted average)
        profile = np.mean(vectors, axis=0)
        return profile

    def top_genres(self, genre_vec: np.ndarray, top_n: int = 5) -> list[str]:
        """Return the genre names with the highest weights."""
        indices = np.argsort(genre_vec)[::-1][:top_n]
        return [ALL_GENRES[i] for i in indices if genre_vec[i] > 0]

    # ── Profile update (incremental) ──────────────────────────────────────────

    def update_with_new_watch(
        self,
        current_genre_vec: np.ndarray,
        new_movie_id: int,
        decay: float = 0.05,
    ) -> np.ndarray:
        """
        Incrementally update a user's genre vector with one new movie.
        Instead of rebuilding from scratch, we nudge the vector slightly.
        This is O(1) — efficient for real-time updates.

        decay=0.05 means the new movie shifts the profile by ~5%.
        """
        if new_movie_id not in self.movies.index:
            return current_genre_vec

        genres_str = str(self.movies.loc[new_movie_id, "genres"])
        g_list = genres_str.split("|") if genres_str and genres_str != "nan" else []
        new_vec = genre_vector(g_list)

        updated = (1 - decay) * current_genre_vec + decay * new_vec

        # Re-normalise
        total = updated.sum()
        if total > 0:
            updated /= total

        return updated

    # ── Full profile summary ──────────────────────────────────────────────────

    def build_full_profile(self, watched_movie_ids: list[int]) -> dict:
        """
        Build a complete user profile snapshot.
        Returns a dict ready to store in DB or return via API.
        """
        genre_vec   = self.build_genre_vector(watched_movie_ids)
        tfidf_vec   = self.build_tfidf_vector(watched_movie_ids)
        top_genres  = self.top_genres(genre_vec)
        total_watch = len(watched_movie_ids)

        return {
            "genre_vector":    genre_vec.tolist(),     # store as JSON in DB
            "top_genres":      top_genres,
            "total_watched":   total_watch,
            "tfidf_available": tfidf_vec is not None,
            # tfidf_vector is NOT stored in DB (too large) — recomputed on demand
        }
