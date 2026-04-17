"""
scripts/train.py — One-command ML training pipeline

Run this ONCE after downloading MovieLens data:
    python scripts/train.py

What it does:
  1. Loads raw MovieLens CSVs from data/
  2. Preprocesses & builds tag soups
  3. Optionally enriches with TMDB (if API key set)
  4. Trains TF-IDF content-based model → saves to artifacts/
  5. Trains SVD collaborative model   → saves to artifacts/
  6. Saves processed CSVs to data/    (fast reload next time)
  7. Runs offline evaluation and prints metrics
"""

import sys
import os
import time
import asyncio
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, ARTIFACTS_DIR
from backend.utils.preprocessing import run_pipeline
from backend.models.content_based import ContentBasedRecommender
from backend.models.collaborative import CollaborativeRecommender
from backend.utils.evaluation import evaluate_recommender


async def enrich_with_tmdb(movies_df: pd.DataFrame, sample_size: int = 500) -> pd.DataFrame:
    """
    Enrich a sample of movies with TMDB data (cast, director, keywords, posters).
    We sample because TMDB has rate limits on the free tier.
    """
    from config import TMDB_API_KEY
    from backend.services.tmdb_service import TMDBService

    if not TMDB_API_KEY:
        print("WARNING: TMDB_API_KEY not set — skipping enrichment.")
        print("Set it in .env to get posters, cast, and keywords.\n")
        return movies_df

    print(f"\nEnriching {sample_size} movies via TMDB API...")
    tmdb    = TMDBService()
    df      = movies_df.copy()
    sample  = df.head(sample_size)
    enriched = []

    for _, row in sample.iterrows():
        data = await tmdb.enrich_movie(row["title"], row.get("year"))
        enriched.append({
            "id": row["id"],
            **data,
        })
        await asyncio.sleep(0.25)  # respect rate limits (~4 req/sec)

    # Merge enrichment back
    enrich_df = pd.DataFrame(enriched).set_index("id")
    for col in ["overview", "cast", "director", "keywords", "poster_path",
                "tmdb_id", "vote_average", "popularity"]:
        if col in enrich_df.columns:
            df.loc[df["id"].isin(enrich_df.index), col] = (
                df["id"].map(enrich_df[col])
            )

    # Rebuild tag soups with the new TMDB data
    from backend.utils.preprocessing import build_tag_soup
    df["tag_soup"] = df.apply(build_tag_soup, axis=1)

    await tmdb.close()
    print(f"TMDB enrichment done for {len(enriched)} movies.\n")
    return df


def train_content_based(movies_df: pd.DataFrame) -> ContentBasedRecommender:
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("STEP 1 — Training Content-Based Model (TF-IDF)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    model = ContentBasedRecommender(str(ARTIFACTS_DIR))
    t0 = time.time()
    model.fit(movies_df)
    model.save()
    print(f"⏱   Took {time.time()-t0:.1f}s\n")
    return model


def train_collaborative(ratings_df: pd.DataFrame) -> CollaborativeRecommender:
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("STEP 2 — Training Collaborative Filter (SVD)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    model = CollaborativeRecommender(str(ARTIFACTS_DIR))
    t0 = time.time()
    # Use a sample for faster training during dev
    sample_ratings = ratings_df.sample(min(500_000, len(ratings_df)), random_state=42)
    metrics = model.fit(sample_ratings, n_epochs=20)
    model.save()
    if metrics:
        print(f"📊  RMSE: {metrics['rmse']:.4f}  MAE: {metrics['mae']:.4f}")
    print(f"⏱   Took {time.time()-t0:.1f}s\n")
    return model


def run_evaluation(content_model, movies_df, ratings_df):
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("STEP 3 — Offline Evaluation")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def recommend_fn(user_id, watched_ids):
        """Wrapper to feed into evaluation harness."""
        if not watched_ids:
            return []
        recs = content_model.recommend_by_movie(
            movie_id=watched_ids[-1], top_n=10, exclude_ids=watched_ids
        )
        return recs

    metrics = evaluate_recommender(
        recommend_fn=recommend_fn,
        ratings_df=ratings_df,
        k=10,
        n_users=200,
    )

        print("\nEvaluation Results:")
    print(f"   Precision@10  : {metrics.get('precision@10', 0):.4f}  (target: ≥ 0.85)")
    print(f"   Recall@10     : {metrics.get('recall@10',    0):.4f}")
    print(f"   NDCG@10       : {metrics.get('ndcg@10',      0):.4f}")
    print(f"   Users tested  : {metrics.get('n_users_evaluated', 0)}")

    p10 = metrics.get("precision@10", 0)
    if p10 >= 0.85:
        print("\nTarget achieved! Precision@10 >= 85%")
    else:
        print(f"\nWARNING: Precision@10 = {p10:.2%} — try more TMDB enrichment or larger dataset.")

    return metrics


async def main():
    print("\nCineMatch — Model Training Pipeline")
    print("=" * 50)

    # ── Check for MovieLens data ───────────────────────────────────────────────
    movies_csv  = DATA_DIR / "movies.csv"
    ratings_csv = DATA_DIR / "ratings.csv"

    if not movies_csv.exists():
        print("\n❌  MovieLens data not found!")
        print("    Download it from: https://grouplens.org/datasets/movielens/25m/")
        print(f"    Place movies.csv and ratings.csv in: {DATA_DIR}/")
        print("\n    Quick start (small dataset for testing):")
        print("    https://grouplens.org/datasets/movielens/latest/  → ml-latest-small.zip")
        return

    # ── Step 0: Preprocess ────────────────────────────────────────────────────
    proc_movies  = DATA_DIR / "processed_movies.csv"
    proc_ratings = DATA_DIR / "processed_ratings.csv"

    if proc_movies.exists() and proc_ratings.exists():
        print("\nLoading cached processed data...")
        movies_df  = pd.read_csv(proc_movies)
        ratings_df = pd.read_csv(proc_ratings)
    else:
        print("\nRunning preprocessing pipeline...")
        movies_df, ratings_df = run_pipeline(DATA_DIR)
        # MovieLens → project format
        if "movieId" in movies_df.columns:
            movies_df = movies_df.rename(columns={"movieId": "id"})
        if "id" in ratings_df.columns and "movieId" not in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={"id": "movieId"})

        # Optional TMDB enrichment (adds poster_path, tmdb_id, etc. when TMDB_API_KEY is set)
        movies_df = await enrich_with_tmdb(movies_df, sample_size=1000)

        # Save processed data so next run is instant
        movies_df.to_csv(proc_movies, index=False)
        ratings_df.to_csv(proc_ratings, index=False)
        print(f"Processed data saved to {DATA_DIR}\n")

    print(f"Dataset: {len(movies_df):,} movies | {len(ratings_df):,} ratings")

    # ── Step 1: Content-based ─────────────────────────────────────────────────
    content_model = train_content_based(movies_df)

    # ── Step 2: Collaborative ─────────────────────────────────────────────────
    collab_model  = train_collaborative(ratings_df)

    # ── Step 3: Evaluation ────────────────────────────────────────────────────
    run_evaluation(content_model, movies_df, ratings_df)

    print("\n" + "=" * 50)
    print("Training complete! Now run the server:")
    print("    uvicorn backend.app:app --reload --port 8000")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
