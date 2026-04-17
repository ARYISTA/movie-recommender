"""
config.py — Central configuration for Movie Recommender System
All environment-based settings live here so changing them is easy.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"   # saved ML models / matrices

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{BASE_DIR}/db.sqlite3")

# ── TMDB ───────────────────────────────────────────────────────────────────────
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# ── Recommendation engine ──────────────────────────────────────────────────────
TOP_N_RECOMMENDATIONS = 10          # default results per request
CONTENT_WEIGHT = 0.6                # weight for content-based score in hybrid
COLLAB_WEIGHT = 0.4                 # weight for collaborative score in hybrid
MIN_RATINGS_FOR_COLLAB = 5          # user needs ≥5 ratings to use collaborative
SIMILARITY_THRESHOLD = 0.1          # cosine similarity cutoff

# ── Mood → genre mapping ───────────────────────────────────────────────────────
MOOD_GENRE_MAP = {
    "happy":     ["Comedy", "Animation", "Family", "Musical"],
    "sad":       ["Drama", "Romance"],
    "excited":   ["Action", "Adventure", "Thriller", "Sci-Fi"],
    "scared":    ["Horror", "Thriller", "Mystery"],
    "romantic":  ["Romance", "Drama"],
    "curious":   ["Documentary", "Biography", "History", "Mystery"],
    "relaxed":   ["Animation", "Family", "Comedy", "Fantasy"],
    "inspired":  ["Biography", "Drama", "Sports", "History"],
}

# ── Time-of-day → mood preference ─────────────────────────────────────────────
# Maps hour ranges to a preferred mood boost (lighter at night, action midday, etc.)
TIME_MOOD_BOOST = {
    range(6, 12):  "relaxed",     # morning
    range(12, 18): "excited",     # afternoon
    range(18, 22): "happy",       # evening
    range(22, 24): "romantic",    # late night
    range(0, 6):   "curious",     # midnight
}

# ── Cold-start ─────────────────────────────────────────────────────────────────
COLD_START_TOP_POPULAR = 20         # fetch N popular movies for new users

# ── App ────────────────────────────────────────────────────────────────────────
APP_TITLE = "Movie Recommender API"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
