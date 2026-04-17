"""
models/hybrid.py — Hybrid Recommendation Engine

HOW IT WORKS:
────────────────────────────────────────────────────────────
We combine content-based + collaborative scores with a weighted sum,
then apply CONTEXT boosts based on:
  • User's mood         → boost genre-matching movies
  • Time of day         → adjust mood automatically
  • Movie popularity    → slight boost for well-known movies
  • Recency             → newer movies get a small lift

Final score = α × content_score + β × collab_score + context_boost

This hybrid approach solves two big problems:
  • Content-only: can't discover new styles (filter bubble)
  • Collaborative-only: fails for new users (cold start)

With the hybrid, we fall back to content-based for new users
and blend in collaborative once they have ≥5 ratings.
────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import (
    CONTENT_WEIGHT, COLLAB_WEIGHT,
    MOOD_GENRE_MAP, TIME_MOOD_BOOST,
    MIN_RATINGS_FOR_COLLAB, TOP_N_RECOMMENDATIONS,
)
from backend.models.content_based import ContentBasedRecommender
from backend.models.collaborative import CollaborativeRecommender


class HybridRecommender:
    """
    Orchestrates content-based + collaborative models and
    applies mood / time-of-day context scoring.
    """

    def __init__(
        self,
        content_model: ContentBasedRecommender,
        collab_model:  CollaborativeRecommender,
        movies_df: pd.DataFrame,          # full movie catalogue
    ):
        self.content = content_model
        self.collab  = collab_model
        self.movies  = movies_df.set_index("id")  # fast lookup by movie_id

    # ── Context helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_time_mood() -> str | None:
        """Return mood string based on current hour, or None."""
        hour = datetime.now().hour
        for hour_range, mood in TIME_MOOD_BOOST.items():
            if hour in hour_range:
                return mood
        return None

    def _genre_boost(self, movie_id: int, preferred_genres: list[str]) -> float:
        """
        Returns 0.0–0.15 bonus if movie's genres overlap with preferred_genres.
        Capped at 0.15 so it doesn't overpower the main score.
        """
        if movie_id not in self.movies.index:
            return 0.0
        movie_genres = str(self.movies.loc[movie_id, "genres"]).split("|")
        overlap = len(set(movie_genres) & set(preferred_genres))
        return min(0.15, overlap * 0.05)

    def _popularity_boost(self, movie_id: int) -> float:
        """Small boost for more popular movies (max +0.05)."""
        if movie_id not in self.movies.index:
            return 0.0
        pop = self.movies.loc[movie_id].get("popularity", 0)
        if pd.isna(pop):
            return 0.0
        # Log-scale normalise: most popular ~1000, typical ~10
        return min(0.05, np.log1p(float(pop)) / 200)

    # ── Main recommend method ─────────────────────────────────────────────────

    def recommend(
        self,
        user_id: int,
        watched_ids: list[int],
        n_ratings: int = 0,
        mood: str | None = None,
        preferred_genres: list[str] | None = None,
        top_n: int = TOP_N_RECOMMENDATIONS,
        user_tfidf_vector=None,         # aggregated TF-IDF profile
    ) -> list[dict]:
        """
        Generate hybrid recommendations.

        Args:
            user_id:            Internal user ID
            watched_ids:        Movies already watched (to exclude)
            n_ratings:          How many explicit ratings user has given
            mood:               e.g. 'happy', 'excited', 'sad' (optional)
            preferred_genres:   Explicit genre preferences (optional)
            top_n:              How many results to return
            user_tfidf_vector:  Pre-computed TF-IDF user profile vector

        Returns:
            Ranked list of dicts with movie info + explanation
        """

        # ── 1. Determine effective mood & preferred genres ─────────────────
        effective_mood = mood or self._get_time_mood()
        if effective_mood and not preferred_genres:
            preferred_genres = MOOD_GENRE_MAP.get(effective_mood, [])

        # ── 2. Get content-based candidates ───────────────────────────────
        if user_tfidf_vector is not None:
            # We have a full user profile vector → use it directly
            content_recs = self.content.recommend_by_profile(
                preference_vector=user_tfidf_vector,
                top_n=top_n * 3,
                exclude_ids=watched_ids,
            )
        elif watched_ids:
            # Fallback: aggregate content scores from recent watches
            all_scores: dict[int, float] = {}
            for wid in watched_ids[-5:]:  # use 5 most recent
                recs = self.content.recommend_by_movie(
                    movie_id=wid,
                    top_n=top_n * 3,
                    exclude_ids=watched_ids,
                )
                for r in recs:
                    mid = r["movie_id"]
                    all_scores[mid] = max(all_scores.get(mid, 0), r["score"])
            content_recs = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            content_recs = [{"movie_id": k, "score": v} for k, v in content_recs]
        else:
            content_recs = []

        # ── 3. Get collaborative candidates (only if enough ratings) ───────
        use_collab = (n_ratings >= MIN_RATINGS_FOR_COLLAB and
                      self.collab.model is not None)

        if use_collab:
            collab_recs = self.collab.recommend(
                user_id=user_id,
                top_n=top_n * 3,
                exclude_ids=watched_ids,
            )
        else:
            collab_recs = []

        # ── 4. Build a unified score map ───────────────────────────────────
        score_map: dict[int, dict] = {}

        for r in content_recs:
            mid = r["movie_id"]
            score_map[mid] = {
                "content_score": r["score"],
                "collab_score":  0.0,
            }

        for r in collab_recs:
            mid = r["movie_id"]
            if mid not in score_map:
                score_map[mid] = {"content_score": 0.0, "collab_score": 0.0}
            score_map[mid]["collab_score"] = r["score"]

        # ── 5. Compute final hybrid score with context boosts ──────────────
        α = CONTENT_WEIGHT if not use_collab else CONTENT_WEIGHT
        β = 0.0            if not use_collab else COLLAB_WEIGHT

        results = []
        for movie_id, scores in score_map.items():
            base = α * scores["content_score"] + β * scores["collab_score"]
            genre_b = self._genre_boost(movie_id, preferred_genres or [])
            pop_b   = self._popularity_boost(movie_id)
            final   = base + genre_b + pop_b

            results.append({
                "movie_id":      movie_id,
                "final_score":   round(final, 4),
                "content_score": round(scores["content_score"], 4),
                "collab_score":  round(scores["collab_score"],  4),
                "genre_boost":   round(genre_b, 4),
                "explanation":   self._build_explanation(
                    movie_id, watched_ids, effective_mood, preferred_genres
                ),
            })

        # ── 6. Sort and return top N ───────────────────────────────────────
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_n]

    # ── Explainability ────────────────────────────────────────────────────────

    def _build_explanation(
        self,
        movie_id: int,
        watched_ids: list[int],
        mood: str | None,
        preferred_genres: list[str] | None,
    ) -> str:
        """
        Generate a human-readable reason for each recommendation.
        e.g. "Because you watched Inception · Matches your excited mood"
        """
        parts = []

        # Seed movie reference
        if watched_ids:
            try:
                seed_title = self.movies.loc[watched_ids[-1], "title"]
                parts.append(f"Because you watched {seed_title}")
            except (KeyError, IndexError):
                pass

        # Genre match
        if preferred_genres and movie_id in self.movies.index:
            movie_genres = str(self.movies.loc[movie_id, "genres"]).split("|")
            matched = list(set(movie_genres) & set(preferred_genres))
            if matched:
                parts.append(f"Matches your {matched[0]} preference")

        # Mood
        if mood:
            parts.append(f"Great for a {mood} mood")

        return " · ".join(parts) if parts else "Highly rated in your taste profile"
