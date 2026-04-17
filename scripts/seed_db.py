"""
scripts/seed_db.py — Populate the SQLite database from processed MovieLens data.

Run AFTER train.py:
    python scripts/seed_db.py

This inserts all movies from processed_movies.csv into the 'movies' table
so the API can do full-text search and return movie details by ID.
It also inserts a sample of ratings into the 'ratings' table for
collaborative filtering lookups at inference time.
"""

import sys
import os
import asyncio
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR
from backend.database.db import init_db, AsyncSessionLocal
from backend.database.models import Movie, User, Rating


BATCH_SIZE = 500   # insert N rows per DB transaction (prevents huge commits)


async def seed_movies(session: AsyncSession, movies_df: pd.DataFrame) -> None:
    """Insert all movies into the DB in batches."""
    print(f"🎬  Seeding {len(movies_df):,} movies...")
    batch = []

    for i, row in movies_df.iterrows():
        movie = Movie(
            id           = int(row["id"]),
            title        = str(row.get("title", ""))[:255],
            genres       = str(row.get("genres", ""))[:500],
            year         = int(row["year"]) if pd.notna(row.get("year")) else None,
            overview     = str(row.get("overview", ""))  if pd.notna(row.get("overview")) else None,
            cast         = str(row.get("cast", ""))[:500] if pd.notna(row.get("cast")) else None,
            director     = str(row.get("director", ""))[:200] if pd.notna(row.get("director")) else None,
            keywords     = str(row.get("keywords", ""))[:500] if pd.notna(row.get("keywords")) else None,
            tmdb_id      = int(row["tmdb_id"]) if pd.notna(row.get("tmdb_id")) else None,
            poster_path  = str(row.get("poster_path", "")) if pd.notna(row.get("poster_path")) else None,
            vote_average = float(row["vote_average"]) if pd.notna(row.get("vote_average")) else 0.0,
            popularity   = float(row["popularity"]) if pd.notna(row.get("popularity")) else 0.0,
            tag_soup     = str(row.get("tag_soup", ""))[:5000] if pd.notna(row.get("tag_soup")) else None,
        )
        batch.append(movie)

        if len(batch) >= BATCH_SIZE:
            session.add_all(batch)
            await session.flush()
            batch = []
            print(f"    inserted {i+1:,}/{len(movies_df):,}...")

    if batch:
        session.add_all(batch)
        await session.flush()

    print("✅  Movies seeded.\n")


async def seed_demo_users(session: AsyncSession) -> None:
    """Create a few demo users so you can test immediately."""
    demo_users = [
        User(id=1, username="alice",  email="alice@demo.com"),
        User(id=2, username="bob",    email="bob@demo.com"),
        User(id=3, username="cinema", email="cinema@demo.com"),
    ]
    for u in demo_users:
        existing = await session.get(User, u.id)
        if not existing:
            session.add(u)

    await session.flush()
    print("✅  Demo users created: alice, bob, cinema\n")


async def seed_sample_ratings(
    session: AsyncSession, ratings_df: pd.DataFrame, n_users: int = 50
) -> None:
    """
    Seed a sample of real MovieLens ratings so collaborative filtering
    has data to work with from day one.
    We import real user IDs 1–n_users from the MovieLens dataset.
    """
    print(f"⭐  Seeding ratings for {n_users} sample users...")
    sample = ratings_df[ratings_df["userId"] <= n_users]
    print(f"    {len(sample):,} rating rows...")

    # Ensure those users exist in our DB
    unique_users = sample["userId"].unique()
    for uid in unique_users:
        existing = await session.get(User, int(uid))
        if not existing:
            session.add(User(id=int(uid), username=f"user_{uid}"))

    await session.flush()

    # Insert ratings in batches
    batch = []
    for i, row in sample.iterrows():
        batch.append(Rating(
            user_id  = int(row["userId"]),
            movie_id = int(row["movieId"]),
            score    = float(row["rating"]),
        ))
        if len(batch) >= BATCH_SIZE:
            session.add_all(batch)
            await session.flush()
            batch = []

    if batch:
        session.add_all(batch)
        await session.flush()

    print("✅  Ratings seeded.\n")


async def main():
    print("\n🗄   CineMatch — Database Seeding")
    print("=" * 40)

    # Load processed data
    proc_movies  = DATA_DIR / "processed_movies.csv"
    proc_ratings = DATA_DIR / "processed_ratings.csv"

    if not proc_movies.exists():
        print("❌  processed_movies.csv not found — run scripts/train.py first.")
        return

    movies_df  = pd.read_csv(proc_movies)
    ratings_df = pd.read_csv(proc_ratings) if proc_ratings.exists() else pd.DataFrame()

    # Initialise tables
    await init_db()

    async with AsyncSessionLocal() as session:
        async with session.begin():
            await seed_movies(session, movies_df)
            await seed_demo_users(session)
            if not ratings_df.empty:
                await seed_sample_ratings(session, ratings_df)

    print("=" * 40)
    print("✅  Database seeded! You can now start the server:")
    print("    uvicorn backend.app:app --reload --port 8000\n")


if __name__ == "__main__":
    asyncio.run(main())
