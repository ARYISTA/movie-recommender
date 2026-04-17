"""
utils/preprocessing.py — Data cleaning + feature engineering for MovieLens dataset.

KEY CONCEPT — "tag soup":
We combine every text feature (genres, cast, director, keywords, overview)
into ONE long string per movie. TF-IDF then converts that string into a
numerical vector. Cosine similarity between vectors = content similarity.

Example tag soup for Toy Story:
  "animation comedy family adventure pixar john_lasseter tom_hanks
   tim_allen don_rickles toys friendship loyalty growing_up"
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load raw MovieLens CSVs
# ─────────────────────────────────────────────────────────────────────────────

def load_movielens(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load movies.csv and ratings.csv from MovieLens dataset folder.
    Download from: https://grouplens.org/datasets/movielens/25m/
    Returns (movies_df, ratings_df).
    """
    data_dir = Path(data_dir)
    movies  = pd.read_csv(data_dir / "movies.csv")   # movieId, title, genres
    ratings = pd.read_csv(data_dir / "ratings.csv")  # userId, movieId, rating, timestamp

    print(f"📦 Loaded {len(movies):,} movies | {len(ratings):,} ratings")
    return movies, ratings


# ─────────────────────────────────────────────────────────────────────────────
# 2. Clean movie titles and extract release year
# ─────────────────────────────────────────────────────────────────────────────

def extract_year(title: str) -> tuple[str, int | None]:
    """
    MovieLens titles look like: 'Toy Story (1995)'
    Returns ('Toy Story', 1995)
    """
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match:
        year  = int(match.group(1))
        clean = title[:match.start()].strip()
        return clean, year
    return title.strip(), None


def clean_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    """Extract year, clean title, split genres into list."""
    df = movies_df.copy()

    # Split combined title+year
    df[["title_clean", "year"]] = df["title"].apply(
        lambda t: pd.Series(extract_year(t))
    )

    # Genres come as "Action|Comedy|Thriller" — split to list
    df["genres_list"] = df["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )

    # Replace original title with clean version
    df["title"] = df["title_clean"]
    df.drop(columns=["title_clean"], inplace=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build "tag soup" — the single feature string per movie
# ─────────────────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """
    Lowercase and replace spaces with underscores so 'John Lasseter'
    becomes one token 'john_lasseter' (not 'john' + 'lasseter').
    """
    return re.sub(r"\s+", "_", str(text).lower().strip())


def build_tag_soup(row: pd.Series) -> str:
    """
    Combine all text features into one token string for TF-IDF.
    We repeat genres 3× to give them more weight in similarity.
    """
    parts = []

    # Genres (repeated for emphasis)
    genres = row.get("genres_list", [])
    for g in genres:
        parts.extend([_slugify(g)] * 3)

    # Director (repeat 2× — director is a strong signal)
    director = row.get("director", "")
    if director and str(director) != "nan":
        parts.extend([_slugify(director)] * 2)

    # Cast (top 3 actors)
    cast = row.get("cast", "")
    if cast and str(cast) != "nan":
        for actor in str(cast).split("|")[:3]:
            parts.append(_slugify(actor))

    # Keywords
    keywords = row.get("keywords", "")
    if keywords and str(keywords) != "nan":
        for kw in str(keywords).split("|")[:10]:
            parts.append(_slugify(kw))

    # Overview (bag-of-words, just append raw)
    overview = row.get("overview", "")
    if overview and str(overview) != "nan":
        # Basic cleaning: lower, remove punctuation
        overview_clean = re.sub(r"[^a-z\s]", "", overview.lower())
        parts.append(overview_clean)

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 4. One-hot encode genres for user profile vector
# ─────────────────────────────────────────────────────────────────────────────

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "Western",
    "Family", "History", "Biography", "Sports", "War",
]


def genre_vector(genres_list: list[str]) -> np.ndarray:
    """
    Convert a list of genre strings to a fixed-size binary vector.
    e.g. ["Action", "Comedy"] → [1, 0, 0, 1, 0, ...]
    """
    vec = np.zeros(len(ALL_GENRES), dtype=np.float32)
    for genre in genres_list:
        if genre in ALL_GENRES:
            vec[ALL_GENRES.index(genre)] = 1.0
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# 5. Filter ratings for meaningful data
# ─────────────────────────────────────────────────────────────────────────────

def filter_ratings(ratings_df: pd.DataFrame,
                   min_user_ratings: int = 20,
                   min_movie_ratings: int = 5) -> pd.DataFrame:
    """
    Remove users who rated fewer than min_user_ratings movies,
    and movies that received fewer than min_movie_ratings ratings.
    This improves collaborative filtering quality significantly.
    """
    df = ratings_df.copy()

    # Keep active users
    user_counts = df.groupby("userId")["rating"].count()
    active_users = user_counts[user_counts >= min_user_ratings].index
    df = df[df["userId"].isin(active_users)]

    # Keep popular enough movies
    movie_counts = df.groupby("movieId")["rating"].count()
    popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
    df = df[df["movieId"].isin(popular_movies)]

    print(f"🔍 Filtered: {df['userId'].nunique():,} users | {df['movieId'].nunique():,} movies")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full pipeline: raw CSVs → clean DataFrames ready for modelling
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Master preprocessing pipeline. Returns (movies_df, ratings_df).
    Call this once, then save results to data/processed/.
    """
    movies, ratings = load_movielens(data_dir)
    movies = clean_movies(movies)
    ratings = filter_ratings(ratings)

    # Build tag soup (uses genres_list; other columns optional)
    movies["tag_soup"] = movies.apply(build_tag_soup, axis=1)

    # Map genres list → pipe-separated string for DB storage
    movies["genres"] = movies["genres_list"].apply("|".join)

    print("Preprocessing complete.")
    return movies, ratings
