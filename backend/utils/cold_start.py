"""
utils/cold_start.py — Handling New Users with No Watch History

The "cold start" problem: a new user has no history, so we can't
personalise recommendations. Our solution: serve trending/popular movies
grouped by genre, then ask the user to rate a few so we can bootstrap their profile.
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import COLD_START_TOP_POPULAR


def get_popular_movies(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    genre_filter: list[str] | None = None,
    top_n: int = COLD_START_TOP_POPULAR,
) -> list[dict]:
    """
    Return top N movies by weighted popularity score.

    We use IMDB's weighted rating formula:
      WR = (v / (v + m)) × R + (m / (v + m)) × C
    where:
      v = number of votes for the movie
      m = minimum votes required (we use 50th percentile)
      R = average rating for the movie
      C = mean rating across all movies
    """
    # Support multiple ratings schemas (MovieLens vs processed exports).
    movie_id_col = "movieId" if "movieId" in ratings_df.columns else ("id" if "id" in ratings_df.columns else None)
    rating_col = "rating" if "rating" in ratings_df.columns else ("score" if "score" in ratings_df.columns else None)
    if movie_id_col is None or rating_col is None or ratings_df.empty:
        return []

    # Compute per-movie stats from ratings
    movie_stats = ratings_df.groupby(movie_id_col).agg(
        vote_count=(rating_col, "count"),
        avg_rating=(rating_col, "mean"),
    ).reset_index()

    # Weighted rating parameters
    C = movie_stats["avg_rating"].mean()
    m = movie_stats["vote_count"].quantile(0.50)

    movie_stats["weighted_score"] = (
        (movie_stats["vote_count"] / (movie_stats["vote_count"] + m)) * movie_stats["avg_rating"] +
        (m                         / (movie_stats["vote_count"] + m)) * C
    )

    # Merge with movie metadata
    # Older processed movie exports may not include poster_path / vote_average yet.
    movies_meta = movies_df.copy()
    if "poster_path" not in movies_meta.columns:
        movies_meta["poster_path"] = ""
    if "vote_average" not in movies_meta.columns:
        movies_meta["vote_average"] = 0.0

    merged = movie_stats.merge(
        movies_meta[["id", "title", "genres", "year", "poster_path", "vote_average"]],
        left_on=movie_id_col,
        right_on="id",
        how="inner",
    )

    # Optional genre filter
    if genre_filter:
        mask = merged["genres"].apply(
            lambda g: any(gen in str(g).split("|") for gen in genre_filter)
        )
        merged = merged[mask]

    # Return top N
    top = merged.nlargest(top_n, "weighted_score")
    return top[["id", "title", "genres", "year", "weighted_score", "poster_path"]].to_dict("records")


def get_onboarding_movies(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    n_per_genre: int = 2,
) -> list[dict]:
    """
    Return a diverse set of movies across genres for new-user onboarding.
    Asking users to rate these gives us an initial preference signal.
    """
    genres_sample = [
        "Action", "Comedy", "Drama", "Sci-Fi", "Romance",
        "Thriller", "Animation", "Documentary", "Horror", "Fantasy"
    ]

    popular = get_popular_movies(movies_df, ratings_df, top_n=500)
    popular_df = pd.DataFrame(popular)

    results = []
    seen_ids = set()

    for genre in genres_sample:
        mask = popular_df["genres"].apply(lambda g: genre in str(g).split("|"))
        genre_movies = popular_df[mask].head(n_per_genre)
        for _, row in genre_movies.iterrows():
            if row["id"] not in seen_ids:
                results.append(row.to_dict())
                seen_ids.add(row["id"])

    return results
