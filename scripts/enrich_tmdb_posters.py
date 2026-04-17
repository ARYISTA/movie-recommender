"""
scripts/enrich_tmdb_posters.py — Backfill TMDB posters into the existing DB.

Why this exists:
- The local MovieLens dataset does not include posters.
- Your current processed_movies.csv also has no poster_path / tmdb_id columns.
- The frontend/backend poster rendering is fine, but the data is missing.

What this script does:
- For movies in the SQLite DB with missing poster_path, call TMDB search by title/year
- Store poster_path (full URL), tmdb_id, overview, vote_average, popularity

Usage (Windows PowerShell):
  .\venv\Scripts\python.exe scripts\enrich_tmdb_posters.py --limit 500

Notes:
- Requires TMDB_API_KEY in .env (see .env.example)
- Respects basic free-tier rate limits by sleeping between requests
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select

from backend.database.db import AsyncSessionLocal
from backend.database.models import Movie
from backend.services.tmdb_service import TMDBService
from config import TMDB_API_KEY


def _is_missing(value: str | None) -> bool:
    if value is None:
        return True
    return str(value).strip() == ""


async def enrich(limit: int, sleep_s: float) -> None:
    if not TMDB_API_KEY:
        raise SystemExit(
            "TMDB_API_KEY is not set. Copy .env.example to .env and set TMDB_API_KEY."
        )

    tmdb = TMDBService()
    updated = 0
    scanned = 0

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Movie)
            .where((Movie.poster_path.is_(None)) | (Movie.poster_path == ""))
            .order_by(Movie.id.asc())
            .limit(limit)
        )
        movies = result.scalars().all()

        print(f"Found {len(movies)} movies missing posters (limit={limit}).")

        for m in movies:
            scanned += 1
            title = (m.title or "").strip()
            if not title:
                continue

            match = await tmdb.search_movie(title, m.year)
            if not match:
                await asyncio.sleep(sleep_s)
                continue

            # Poster URL (full URL)
            poster_rel = match.get("poster_path") or ""
            poster_full = f"{tmdb.image_base}{poster_rel}" if poster_rel else ""

            tmdb_id = match.get("id")
            overview = match.get("overview") or ""
            vote_average = float(match.get("vote_average") or 0.0)
            popularity = float(match.get("popularity") or 0.0)

            # Only write fields we can confidently improve.
            changed = False
            if tmdb_id and m.tmdb_id is None:
                m.tmdb_id = int(tmdb_id)
                changed = True
            if not _is_missing(poster_full) and _is_missing(m.poster_path):
                m.poster_path = poster_full
                changed = True
            if overview and _is_missing(m.overview):
                m.overview = overview
                changed = True
            if vote_average and (m.vote_average is None or float(m.vote_average) == 0.0):
                m.vote_average = vote_average
                changed = True
            if popularity and (m.popularity is None or float(m.popularity) == 0.0):
                m.popularity = popularity
                changed = True

            if changed:
                updated += 1

            # Commit in small batches so we can resume safely if interrupted.
            if scanned % 50 == 0:
                await session.commit()
                print(f"Progress: scanned={scanned} updated={updated}")

            await asyncio.sleep(sleep_s)

        await session.commit()

    await tmdb.close()
    print(f"Done. scanned={scanned} updated={updated}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=500, help="Max movies to enrich this run")
    p.add_argument(
        "--sleep",
        type=float,
        default=0.30,
        help="Seconds to sleep between TMDB requests (rate limiting)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(enrich(limit=args.limit, sleep_s=args.sleep))

