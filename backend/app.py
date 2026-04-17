"""
backend/app.py — FastAPI application entry point

Run locally with:
    uvicorn backend.app:app --reload --port 8000

Then visit:
    http://localhost:8000        → Frontend
    http://localhost:8000/docs  → Auto-generated API docs (Swagger UI)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import APP_TITLE, APP_VERSION, DATA_DIR, ARTIFACTS_DIR
from backend.database.db import init_db, AsyncSessionLocal
from backend.database.models import Movie
from backend.api.routes import router
from backend.services.recommendation_service import recommendation_service
from sqlalchemy import select, func


# ── Startup / shutdown lifecycle ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code here runs BEFORE the app starts accepting requests.
    Perfect for loading ML models, DB setup, etc.
    """
    print("Starting Movie Recommender API...")

    # 1. Create database tables
    await init_db()

    # 2. Load movie data (from pre-processed CSV or DB)
    processed_movies  = DATA_DIR / "processed_movies.csv"
    processed_ratings = DATA_DIR / "processed_ratings.csv"

    if processed_movies.exists() and processed_ratings.exists():
        print("Loading processed data from CSV...")
        movies_df  = pd.read_csv(processed_movies)
        ratings_df = pd.read_csv(processed_ratings)
        # Backwards/alternate column names compatibility.
        # Some pipelines output movie id as `id` instead of `movieId`.
        if "movieId" not in ratings_df.columns and "id" in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={"id": "movieId"})
        if "userId" not in ratings_df.columns and "user_id" in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={"user_id": "userId"})
        if "rating" not in ratings_df.columns and "score" in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={"score": "rating"})
    else:
        # No processed data yet — skip ML loading
        # Run: python scripts/train.py to generate processed data
        print("WARNING: No processed data found. Run scripts/train.py first.")
        movies_df  = pd.DataFrame(columns=["id", "title", "genres", "tag_soup"])
        ratings_df = pd.DataFrame(columns=["userId", "movieId", "rating"])

    # 3. Load ML models (or train if not saved)
    if not movies_df.empty:
        recommendation_service.load(movies_df, ratings_df)

        # If the DB hasn't been seeded yet, auto-populate the movies table from the
        # processed CSV so search/trending/top-rated/watch work end-to-end.
        async with AsyncSessionLocal() as session:
            n_movies = await session.scalar(select(func.count(Movie.id))) or 0
            if int(n_movies) == 0:
                print("Movies table empty — seeding from processed_movies.csv...")
                batch = []
                for _, row in movies_df.iterrows():
                    batch.append(
                        Movie(
                            id=int(row["id"]),
                            title=str(row.get("title", ""))[:255],
                            genres=str(row.get("genres", ""))[:500] if pd.notna(row.get("genres")) else None,
                            year=int(row["year"]) if pd.notna(row.get("year")) else None,
                            overview=str(row.get("overview", "")) if pd.notna(row.get("overview")) else None,
                            cast=str(row.get("cast", ""))[:500] if pd.notna(row.get("cast")) else None,
                            director=str(row.get("director", ""))[:200] if pd.notna(row.get("director")) else None,
                            keywords=str(row.get("keywords", ""))[:500] if pd.notna(row.get("keywords")) else None,
                            tmdb_id=int(row["tmdb_id"]) if pd.notna(row.get("tmdb_id")) else None,
                            poster_path=str(row.get("poster_path", "")) if pd.notna(row.get("poster_path")) else None,
                            vote_average=float(row["vote_average"]) if pd.notna(row.get("vote_average")) else 0.0,
                            popularity=float(row["popularity"]) if pd.notna(row.get("popularity")) else 0.0,
                            tag_soup=str(row.get("tag_soup", ""))[:5000] if pd.notna(row.get("tag_soup")) else None,
                        )
                    )
                    if len(batch) >= 750:
                        session.add_all(batch)
                        await session.commit()
                        batch = []
                if batch:
                    session.add_all(batch)
                    await session.commit()
                print("Movies seeded.")

    yield   # App runs here

    # Cleanup on shutdown
    print("Shutting down...")
    from backend.services.tmdb_service import tmdb
    await tmdb.close()


# ── App instance ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="AI-powered movie recommendation system with content-based, collaborative, and hybrid filtering.",
    lifespan=lifespan,
)

# ── CORS (allow frontend on any origin during development) ────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Restrict in production: ["https://your-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount API routes ───────────────────────────────────────────────────────────
app.include_router(router)

# ── Serve frontend static files ───────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        """Serve the main HTML file at the root URL."""
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": APP_VERSION}
