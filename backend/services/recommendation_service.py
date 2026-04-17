"""
services/recommendation_service.py — Orchestrates all models for the API layer.

This is the "brain" that routes requests to the right model,
handles edge cases (cold start, missing data), and formats the
final response the API returns to the frontend.
"""

import numpy as np
import pandas as pd
from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from backend.models.content_based import ContentBasedRecommender
from backend.models.collaborative  import CollaborativeRecommender
from backend.models.hybrid         import HybridRecommender
from backend.models.user_profile   import UserProfileBuilder
from backend.utils.cold_start      import get_popular_movies
from config import TOP_N_RECOMMENDATIONS


class RecommendationService:
    """
    High-level service used by FastAPI routes.
    Loads models once at startup and reuses them for every request.
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.content_model = ContentBasedRecommender(artifacts_dir)
        self.collab_model  = CollaborativeRecommender(artifacts_dir)
        self.hybrid: HybridRecommender | None = None
        self.profile_builder: UserProfileBuilder | None = None
        self.movies_df: pd.DataFrame | None = None
        self.ratings_df: pd.DataFrame | None = None
        self._ready = False

    def load(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> None:
        """
        Load pre-trained models and movie data.
        Called once at FastAPI startup.
        """
        self.movies_df  = movies_df
        self.ratings_df = ratings_df

        # Try to load saved models; if not found, train fresh
        cb_loaded = self.content_model.load()
        cf_loaded = self.collab_model.load()

        if not cb_loaded:
            print("No saved content model — training now (takes ~30s)...")
            self.content_model.fit(movies_df)
            self.content_model.save()

        # Instantiate higher-level objects
        self.hybrid = HybridRecommender(
            content_model=self.content_model,
            collab_model=self.collab_model,
            movies_df=movies_df,
        )
        self.profile_builder = UserProfileBuilder(
            content_model=self.content_model,
            movies_df=movies_df,
        )

        self._ready = True
        print("Recommendation service ready.")

    # ── Core recommendation method ────────────────────────────────────────────

    def get_recommendations(
        self,
        user_id: int,
        watched_ids: list[int],
        n_ratings: int = 0,
        mood: Optional[str] = None,
        genres: Optional[list[str]] = None,
        top_n: int = TOP_N_RECOMMENDATIONS,
    ) -> list[dict]:
        """
        Main entry point for recommendations.
        Automatically handles cold-start users.
        """
        if not self._ready:
            raise RuntimeError("Service not loaded. Call load() first.")

        # ── Cold start: user has no history ──────────────────────────────────
        if not watched_ids:
            popular = get_popular_movies(
                self.movies_df, self.ratings_df,
                genre_filter=genres, top_n=top_n
            )
            return [
                {
                    "movie_id":    m["id"],
                    "final_score": m.get("weighted_score", 0.0),
                    "content_score": 0.0,
                    "collab_score":  0.0,
                    "genre_boost":   0.0,
                    "explanation": "Trending & highly rated",
                }
                for m in popular
            ]

        # ── Build TF-IDF user profile ─────────────────────────────────────────
        tfidf_vec = self.profile_builder.build_tfidf_vector(watched_ids)

        # ── Hybrid recommendations ────────────────────────────────────────────
        recs = self.hybrid.recommend(
            user_id=user_id,
            watched_ids=watched_ids,
            n_ratings=n_ratings,
            mood=mood,
            preferred_genres=genres,
            top_n=top_n,
            user_tfidf_vector=tfidf_vec,
        )

        return recs

    # ── Enrich recs with movie metadata ──────────────────────────────────────

    def enrich(self, recs: list[dict]) -> list[dict]:
        """
        Attach movie metadata (title, genres, poster, year) to each recommendation.
        """
        if self.movies_df is None:
            return recs

        movies_indexed = self.movies_df.set_index("id")
        enriched = []

        for rec in recs:
            mid = rec["movie_id"]
            if mid in movies_indexed.index:
                row = movies_indexed.loc[mid]
                enriched.append({
                    **rec,
                    "title":       row.get("title", "Unknown"),
                    "genres":      row.get("genres", ""),
                    "year":        int(row["year"]) if pd.notna(row.get("year")) else None,
                    "poster_path": row.get("poster_path", ""),
                    "vote_average": float(row.get("vote_average", 0)),
                })
            else:
                enriched.append({**rec, "title": "Unknown", "genres": "", "year": None})

        return enriched

    # ── Build user profile summary ────────────────────────────────────────────

    def get_user_profile(self, watched_ids: list[int]) -> dict:
        """Return a human-readable user profile dict."""
        if not self.profile_builder or not watched_ids:
            return {"top_genres": [], "total_watched": 0}

        genre_vec = self.profile_builder.build_genre_vector(watched_ids)
        top_genres = self.profile_builder.top_genres(genre_vec)

        return {
            "top_genres":    top_genres,
            "total_watched": len(watched_ids),
            "genre_breakdown": {
                genre: round(float(genre_vec[i]), 3)
                for i, genre in enumerate(__import__("backend.utils.preprocessing",
                    fromlist=["ALL_GENRES"]).ALL_GENRES)
                if genre_vec[i] > 0.01
            },
        }


# ── Singleton (loaded at app startup) ─────────────────────────────────────────
recommendation_service = RecommendationService()
