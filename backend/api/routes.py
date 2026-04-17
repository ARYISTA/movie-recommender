"""
api/routes.py — FastAPI route definitions (all endpoints)

Endpoints:
  POST /api/users                  — create a new user
  GET  /api/users/{user_id}/profile — get user profile + top genres
  POST /api/users/{user_id}/watch  — log a watched movie
  POST /api/users/{user_id}/rate   — submit a star rating
  GET  /api/users/{user_id}/history — watch history
  GET  /api/recommend/{user_id}    — get personalised recommendations
  GET  /api/movies/search          — search movies by title
  GET  /api/movies/trending        — trending movies
  GET  /api/movies/top-rated       — top-rated from catalog
  GET  /api/tmdb/movie/{tmdb_id}   — TMDB details (trending modal)
  GET  /api/users/lookup?username= — resolve username → id
  GET  /api/movies/{movie_id}      — single movie details
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from backend.database.db import get_db
from backend.database.models import User, Movie, WatchHistory, Rating
from backend.services.recommendation_service import recommendation_service
from backend.services.tmdb_service import tmdb

router = APIRouter(prefix="/api")


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas — define the shape of request / response bodies
# ══════════════════════════════════════════════════════════════════════════════

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=80)
    email: Optional[str] = None

class WatchRequest(BaseModel):
    movie_id: int
    watch_progress: int = Field(default=100, ge=0, le=100)

class RatingRequest(BaseModel):
    movie_id: int
    score: float = Field(..., ge=0.5, le=5.0)

class RecommendResponse(BaseModel):
    movie_id:      int
    title:         str
    genres:        str
    year:          Optional[int]
    poster_path:   str
    vote_average:  float
    final_score:   float
    explanation:   str


# ══════════════════════════════════════════════════════════════════════════════
# User endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/users", status_code=201)
async def create_user(body: UserCreate, db: AsyncSession = Depends(get_db)):
    """Create a new user account."""
    # Check if username already taken
    existing = await db.scalar(select(User).where(User.username == body.username))
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken.")

    user = User(username=body.username, email=body.email)
    db.add(user)
    await db.flush()   # flush to get the generated ID
    return {"id": user.id, "username": user.username, "message": "User created"}


@router.get("/users/lookup")
async def lookup_user(
    username: str = Query(..., min_length=3, max_length=80),
    db: AsyncSession = Depends(get_db),
):
    """Resolve username → user id (for returning users when POST /users returns 409)."""
    user = await db.scalar(select(User).where(User.username == username))
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"id": user.id, "username": user.username}


@router.get("/users/{user_id}/profile")
async def get_user_profile(user_id: int, db: AsyncSession = Depends(get_db)):
    """Return user info + dynamically computed preference profile."""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Fetch watch history (IDs only)
    result = await db.execute(
        select(WatchHistory.movie_id)
        .where(WatchHistory.user_id == user_id)
        .order_by(desc(WatchHistory.watched_at))
    )
    watched_ids = [row[0] for row in result.all()]
    # Ensure profile stats reflect unique titles watched (not raw events).
    # Keep most-recent ordering for downstream profile building.
    unique_watched_ids = list(dict.fromkeys(watched_ids))

    # Build profile using the ML service
    profile = recommendation_service.get_user_profile(unique_watched_ids)

    return {
        "user_id":   user.id,
        "username":  user.username,
        "joined":    user.created_at.isoformat(),
        **profile,
    }


@router.post("/users/{user_id}/watch")
async def log_watch(user_id: int, body: WatchRequest, db: AsyncSession = Depends(get_db)):
    """Log a movie as watched. Also triggers profile vector update."""
    user  = await db.get(User, user_id)
    movie = await db.get(Movie, body.movie_id)

    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found.")

    # Add to watch history
    entry = WatchHistory(
        user_id=user_id,
        movie_id=body.movie_id,
        watch_progress=body.watch_progress,
    )
    db.add(entry)
    await db.flush()

    return {
        "message":    f"Logged '{movie.title}' for user {user.username}",
        "history_id": entry.id,
    }


@router.post("/users/{user_id}/rate")
async def rate_movie(user_id: int, body: RatingRequest, db: AsyncSession = Depends(get_db)):
    """Submit or update a star rating for a movie."""
    # Check if rating already exists (update it if so)
    existing = await db.scalar(
        select(Rating).where(
            Rating.user_id  == user_id,
            Rating.movie_id == body.movie_id,
        )
    )

    if existing:
        existing.score    = body.score
        existing.rated_at = datetime.utcnow()
    else:
        db.add(Rating(user_id=user_id, movie_id=body.movie_id, score=body.score))

    return {"message": "Rating saved", "score": body.score}


@router.get("/users/{user_id}/history")
async def get_history(
    user_id: int,
    limit: int = Query(default=20, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Return the user's watch history with movie titles."""
    result = await db.execute(
        select(WatchHistory, Movie.title, Movie.genres, Movie.year, Movie.poster_path)
        .join(Movie, WatchHistory.movie_id == Movie.id)
        .where(WatchHistory.user_id == user_id)
        .order_by(desc(WatchHistory.watched_at))
        .limit(limit)
    )

    history = [
        {
            "movie_id":       row.WatchHistory.movie_id,
            "title":          row.title,
            "genres":         row.genres,
            "year":           row.year,
            "poster_path":    row.poster_path,
            "watched_at":     row.WatchHistory.watched_at.isoformat(),
            "watch_progress": row.WatchHistory.watch_progress,
        }
        for row in result.all()
    ]

    # Opportunistic poster backfill for recently watched items (small, rate-limited).
    # This keeps History visually rich without requiring a full offline enrichment run.
    if tmdb.api_key:
        missing = [h for h in history if not (h.get("poster_path") or "").strip()]
        missing = missing[:5]  # keep requests bounded per call
        if missing:
            for h in missing:
                try:
                    match = await tmdb.search_movie(h["title"], h.get("year"))
                    if not match:
                        continue
                    poster_rel = match.get("poster_path") or ""
                    poster_full = f"{tmdb.image_base}{poster_rel}" if poster_rel else ""
                    tmdb_id = match.get("id")

                    if poster_full:
                        movie = await db.get(Movie, int(h["movie_id"]))
                        if movie and (not movie.poster_path or not str(movie.poster_path).strip()):
                            movie.poster_path = poster_full
                        if movie and tmdb_id and movie.tmdb_id is None:
                            movie.tmdb_id = int(tmdb_id)
                        h["poster_path"] = poster_full
                except Exception:
                    # best-effort enrichment; don't fail history on TMDB hiccups
                    continue

    return {"user_id": user_id, "history": history}


# ══════════════════════════════════════════════════════════════════════════════
# Recommendation endpoint
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/recommend/{user_id}", response_model=list[RecommendResponse])
async def get_recommendations(
    user_id: int,
    mood:    Optional[str]       = Query(default=None, description="happy|sad|excited|scared|romantic|curious|relaxed|inspired"),
    genres:  Optional[list[str]] = Query(default=None, description="Genre filters e.g. Action,Comedy"),
    top_n:   int                 = Query(default=10, le=50),
    db: AsyncSession             = Depends(get_db),
):
    """
    Core recommendation endpoint.
    Returns personalised movies ranked by hybrid score.
    """
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Fetch watch history
    result = await db.execute(
        select(WatchHistory.movie_id)
        .where(WatchHistory.user_id == user_id)
        .order_by(WatchHistory.watched_at)   # chronological order matters for profile
    )
    watched_ids = [row[0] for row in result.all()]

    # Count ratings (to decide whether to use collaborative filtering)
    n_ratings = await db.scalar(
        select(func.count(Rating.id)).where(Rating.user_id == user_id)
    ) or 0

    # Get raw recommendations from ML service
    recs = recommendation_service.get_recommendations(
        user_id=user_id,
        watched_ids=watched_ids,
        n_ratings=int(n_ratings),
        mood=mood,
        genres=genres,
        top_n=top_n,
    )

    # Enrich with movie metadata
    enriched = recommendation_service.enrich(recs)

    # IMPORTANT: posters in the DB may be enriched after startup, while the in-memory
    # movies_df used by the recommendation service may not have poster_path populated.
    # Hydrate poster_path (and a couple display fields) from the DB to keep the UI correct.
    ids = [int(m.get("movie_id")) for m in enriched if m.get("movie_id") is not None]
    if ids:
        result2 = await db.execute(select(Movie).where(Movie.id.in_(ids)))
        by_id = {m.id: m for m in result2.scalars().all()}

        for item in enriched:
            mid = int(item["movie_id"])
            m = by_id.get(mid)
            if not m:
                continue
            # Prefer DB values when present (DB is the source of truth after enrichment).
            if m.poster_path and str(m.poster_path).strip():
                item["poster_path"] = m.poster_path
            if m.vote_average and float(m.vote_average) > 0:
                item["vote_average"] = float(m.vote_average)
            if m.year is not None:
                item["year"] = m.year

        # Opportunistic TMDB backfill for missing posters in the returned rec set
        # (bounded to keep latency and rate limits sane).
        if tmdb.api_key:
            missing_items = [it for it in enriched if not (it.get("poster_path") or "").strip()]
            for it in missing_items[:5]:
                try:
                    title = it.get("title") or ""
                    year = it.get("year")
                    match = await tmdb.search_movie(title, year)
                    if not match:
                        continue
                    poster_rel = match.get("poster_path") or ""
                    poster_full = f"{tmdb.image_base}{poster_rel}" if poster_rel else ""
                    tmdb_id = match.get("id")
                    if not poster_full:
                        continue

                    movie = by_id.get(int(it["movie_id"])) or await db.get(Movie, int(it["movie_id"]))
                    if movie and (not movie.poster_path or not str(movie.poster_path).strip()):
                        movie.poster_path = poster_full
                    if movie and tmdb_id and movie.tmdb_id is None:
                        movie.tmdb_id = int(tmdb_id)

                    it["poster_path"] = poster_full
                except Exception:
                    continue

    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# Movie endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/movies/search")
async def search_movies(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, le=50),
    db: AsyncSession = Depends(get_db),
):
    """Full-text search on movie titles."""
    result = await db.execute(
        select(Movie)
        .where(Movie.title.ilike(f"%{q}%"))
        .limit(limit)
    )
    movies = result.scalars().all()
    return [
        {
            "id":          m.id,
            "title":       m.title,
            "genres":      m.genres,
            "year":        m.year,
            "poster_path": m.poster_path,
            "vote_average": m.vote_average,
        }
        for m in movies
    ]


@router.get("/movies/trending")
async def get_trending(
    limit: int = Query(default=20, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch trending movies.
    Prefer TMDB (live). If TMDB isn't configured or fails, fall back to
    a "trending" approximation from our local catalog (popularity/votes).
    """
    movies = await tmdb.get_trending()
    if movies:
        return movies[:limit]

    # Fallback: approximate "trending" from local data.
    # Prefer ratings volume (MovieLens) when available; otherwise, use catalog order.
    ids: list[int] | None = None
    if getattr(recommendation_service, "ratings_df", None) is not None:
        try:
            ratings_df = recommendation_service.ratings_df
            counts = ratings_df.groupby("movieId")["rating"].count().sort_values(ascending=False)
            ids = [int(i) for i in counts.head(limit).index.tolist()]
        except Exception:
            ids = None

    if ids:
        result = await db.execute(select(Movie).where(Movie.id.in_(ids)))
        movies_by_id = {m.id: m for m in result.scalars().all()}
        local = [movies_by_id[i] for i in ids if i in movies_by_id]
    else:
        result = await db.execute(select(Movie).order_by(desc(Movie.popularity), desc(Movie.vote_average)).limit(limit))
        local = result.scalars().all()

    return [
        {
            "id": m.id,
            "title": m.title,
            "genres": m.genres,
            "year": m.year,
            "poster_path": m.poster_path,
            "vote_average": m.vote_average,
            "popularity": m.popularity,
        }
        for m in local
    ]


@router.get("/movies/top-rated")
async def get_top_rated(
    limit: int = Query(default=20, le=50),
    db: AsyncSession = Depends(get_db),
):
    """Top-rated movies from our catalog (by TMDB-style vote_average)."""
    result = await db.execute(
        select(Movie)
        .where(Movie.vote_average.is_not(None))
        .where(Movie.vote_average > 0)
        .order_by(desc(Movie.vote_average), desc(Movie.popularity))
        .limit(limit)
    )
    movies = result.scalars().all()
    if not movies and getattr(recommendation_service, "ratings_df", None) is not None:
        # Fallback: compute top-rated from MovieLens ratings (mean rating scaled to /10).
        try:
            ratings_df = recommendation_service.ratings_df
            means = ratings_df.groupby("movieId")["rating"].mean().sort_values(ascending=False)
            ids = [int(i) for i in means.head(limit).index.tolist()]

            result2 = await db.execute(select(Movie).where(Movie.id.in_(ids)))
            movies_by_id = {m.id: m for m in result2.scalars().all()}
            movies = [movies_by_id[i] for i in ids if i in movies_by_id]
            # Attach computed vote_average (MovieLens rating 0.5–5 → approx 1–10)
            computed = {int(i): float(means.loc[i]) * 2.0 for i in ids if i in means.index}
            return [
                {
                    "id": m.id,
                    "title": m.title,
                    "genres": m.genres,
                    "year": m.year,
                    "poster_path": m.poster_path,
                    "vote_average": computed.get(m.id, 0.0),
                }
                for m in movies
            ]
        except Exception:
            pass
    return [
        {
            "id": m.id,
            "title": m.title,
            "genres": m.genres,
            "year": m.year,
            "poster_path": m.poster_path,
            "vote_average": m.vote_average,
        }
        for m in movies
    ]


@router.get("/tmdb/movie/{tmdb_id}")
async def get_tmdb_movie_for_modal(tmdb_id: int):
    """Full movie payload from TMDB (for trending cards that only have tmdb_id)."""
    if not tmdb.api_key:
        raise HTTPException(
            status_code=503,
            detail="TMDB API key not configured. Set TMDB_API_KEY in .env",
        )
    payload = await tmdb.get_movie_modal_payload(tmdb_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Movie not found on TMDB.")
    return payload


@router.get("/movies/{movie_id}")
async def get_movie(movie_id: int, db: AsyncSession = Depends(get_db)):
    """Get details for a single movie."""
    movie = await db.get(Movie, movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found.")

    return {
        "id":          movie.id,
        "title":       movie.title,
        "genres":      movie.genres_list(),
        "year":        movie.year,
        "overview":    movie.overview,
        "cast":        movie.cast,
        "director":    movie.director,
        "poster_path": movie.poster_path,
        "vote_average": movie.vote_average,
        "popularity":  movie.popularity,
    }
