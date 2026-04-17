#!/bin/bash
# run.sh — Start the CineMatch server (dev mode)
# Usage: bash run.sh

set -e

echo "🎬  CineMatch — Movie Recommendation System"
echo "==========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌  Python 3 not found. Install Python 3.10+"
    exit 1
fi

# Create virtualenv if not exists
if [ ! -d "venv" ]; then
    echo "📦  Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "📦  Installing dependencies..."
pip install -q -r requirements.txt

# Copy .env if not exists
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚙️   Created .env from template. Edit it to add your TMDB key."
fi

# Check for data
if [ ! -f "data/movies.csv" ]; then
    echo ""
    echo "⚠️   MovieLens data not found in data/"
    echo "     Download from: https://grouplens.org/datasets/movielens/latest/"
    echo "     Get ml-latest-small.zip, extract movies.csv and ratings.csv to data/"
    echo ""
fi

# Train if no artifacts
if [ ! -f "artifacts/tfidf_vectorizer.pkl" ]; then
    echo "🧠  Training ML models (first run only)..."
    python scripts/train.py
    echo "🗄   Seeding database..."
    python scripts/seed_db.py
fi

# Start server
echo ""
echo "🚀  Starting server at http://localhost:8000"
echo "📖  API docs at http://localhost:8000/docs"
echo ""

uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
