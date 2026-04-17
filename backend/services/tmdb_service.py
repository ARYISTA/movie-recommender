"""
services/tmdb_service.py — TMDB (The Movie Database) API wrapper

TMDB enriches our MovieLens data with:
  • Movie posters
  • Full cast & crew (director, top actors)
  • Plot overview / synopsis
  • Keywords (thematic tags like "time travel", "heist")
  • Trending movies (for cold-start / homepage)

Get a free API key at: https://www.themoviedb.org/settings/api
Add it to .env as: TMDB_API_KEY=your_key_here
"""

import httpx
import asyncio
from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import TMDB_API_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE


class TMDBService:
    """Async TMDB API client using httpx."""

    def __init__(self):
        self.api_key  = TMDB_API_KEY
        self.base_url = TMDB_BASE_URL
        self.image_base = TMDB_IMAGE_BASE
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()

    # ── Internal request helper ───────────────────────────────────────────────

    async def _get(self, endpoint: str, params: dict | None = None) -> dict | None:
        """Make an authenticated GET request to TMDB."""
        if not self.api_key:
            return None  # API key not configured

        client = await self._get_client()
        base_params = {"api_key": self.api_key}
        if params:
            base_params.update(params)

        try:
            response = await client.get(f"{self.base_url}/{endpoint}", params=base_params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"WARNING: TMDB request failed: {e}")
            return None

    # ── Public API methods ────────────────────────────────────────────────────

    async def search_movie(self, title: str, year: int | None = None) -> dict | None:
        """Search for a movie by title, return the best match."""
        params = {"query": title, "include_adult": "false"}
        if year:
            params["year"] = year

        data = await self._get("search/movie", params)
        if data and data.get("results"):
            return data["results"][0]
        return None

    async def get_movie_details(self, tmdb_id: int) -> dict | None:
        """Get full movie details including credits and keywords."""
        data = await self._get(
            f"movie/{tmdb_id}",
            {"append_to_response": "credits,keywords"}
        )
        return data

    async def enrich_movie(self, title: str, year: int | None = None) -> dict:
        """
        One-stop shop: search + fetch details + format results.
        Returns a dict with the fields we care about for recommendations.
        """
        result = {}

        # Step 1: find the movie
        match = await self.search_movie(title, year)
        if not match:
            return result

        tmdb_id = match.get("id")
        result["tmdb_id"]     = tmdb_id
        result["overview"]    = match.get("overview", "")
        result["popularity"]  = match.get("popularity", 0.0)
        result["vote_average"] = match.get("vote_average", 0.0)

        # Build poster URL
        poster = match.get("poster_path", "")
        result["poster_path"] = f"{self.image_base}{poster}" if poster else ""

        # Step 2: fetch full details with cast + keywords
        details = await self.get_movie_details(tmdb_id)
        if not details:
            return result

        # Director
        crew = details.get("credits", {}).get("crew", [])
        directors = [p["name"] for p in crew if p.get("job") == "Director"]
        result["director"] = directors[0] if directors else ""

        # Top-3 cast members
        cast = details.get("credits", {}).get("cast", [])
        result["cast"] = "|".join(p["name"] for p in cast[:3])

        # Keywords
        keywords_raw = details.get("keywords", {}).get("keywords", [])
        result["keywords"] = "|".join(k["name"] for k in keywords_raw[:10])

        return result

    async def get_trending(self, time_window: str = "week") -> list[dict]:
        """
        Fetch trending movies from TMDB (day or week).
        Used for: cold-start homepage, trending section.
        """
        data = await self._get(f"trending/movie/{time_window}")
        if not data:
            return []

        movies = []
        for m in data.get("results", []):
            poster = m.get("poster_path", "")
            movies.append({
                "tmdb_id":      m.get("id"),
                "title":        m.get("title"),
                "overview":     m.get("overview", ""),
                "vote_average": m.get("vote_average", 0.0),
                "popularity":   m.get("popularity", 0.0),
                "poster_path":  f"{self.image_base}{poster}" if poster else "",
                "genres":       "",  # genre IDs only from trending endpoint
            })
        return movies

    async def get_movie_modal_payload(self, tmdb_id: int) -> dict | None:
        """
        Format TMDB movie/{id} response for the frontend modal (trending / external IDs).
        """
        d = await self.get_movie_details(tmdb_id)
        if not d:
            return None

        poster = d.get("poster_path") or ""
        poster_path = f"{self.image_base}{poster}" if poster else ""

        release = d.get("release_date") or ""
        year = None
        if len(release) >= 4 and release[:4].isdigit():
            year = int(release[:4])

        crew = d.get("credits", {}).get("crew", [])
        directors = [p["name"] for p in crew if p.get("job") == "Director"]
        director = directors[0] if directors else ""

        cast_list = d.get("credits", {}).get("cast", [])
        cast = "|".join(p["name"] for p in cast_list[:3])

        genres = [g["name"] for g in d.get("genres", [])]

        return {
            "source": "tmdb",
            "tmdb_id": tmdb_id,
            "title": d.get("title", "Unknown"),
            "year": year,
            "genres": genres,
            "overview": d.get("overview", "") or "",
            "director": director,
            "cast": cast,
            "poster_path": poster_path,
            "vote_average": float(d.get("vote_average") or 0.0),
            "popularity": float(d.get("popularity") or 0.0),
        }


# ── Singleton instance ─────────────────────────────────────────────────────────
tmdb = TMDBService()
