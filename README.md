# 🎬 CineMatch — AI Movie Recommendation System

> Personalised movie recommendations using Content-Based Filtering, Collaborative Filtering (SVD), and a Hybrid model — with mood, genre, and time-of-day context awareness.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn)
![SQLite](https://img.shields.io/badge/Database-SQLite%20%2F%20MySQL-lightgrey?logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 What It Does

CineMatch analyses your **watch history**, **mood**, and **preferred genres** to recommend movies you'll love. It uses three complementary ML techniques:

| Model | Technique | When Used |
|-------|-----------|-----------|
| Content-Based | TF-IDF + Cosine Similarity | All users |
| Collaborative | SVD Matrix Factorization | Users with ≥5 ratings |
| Hybrid | Weighted blend + context boosts | Always (final ranker) |

**Key features:**
- 🎭 Mood-based filtering (happy → Comedy, excited → Action, etc.)
- ⏰ Time-of-day automatic mood detection
- 🧑‍💻 Dynamic user profile that evolves with every watch
- ❄️ Cold-start handling (popularity-based for new users)
- 💡 Explainable recommendations ("Because you watched Inception")
- 🌐 TMDB integration for posters, cast, and trending movies
- 📊 Evaluation framework (Precision@10, Recall@10, NDCG@10)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Frontend (HTML/CSS/JS)              │
│   Mood Selector · Genre Filter · Movie Grid          │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────┐
│               FastAPI Backend                        │
│   /recommend · /add-watched · /profile · /trending  │
└──────┬──────────────┬──────────────┬────────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌───▼──────────┐
│  Content    │ │Collaborative│ │  User Profile │
│  TF-IDF     │ │   SVD       │ │  Builder      │
│  Cosine Sim │ │  (Surprise) │ │  Genre Vecs   │
└──────┬──────┘ └─────┬──────┘ └───────────────┘
       └──────────────┘
              │ Hybrid Ranker
┌─────────────▼───────────────┐
│  Mood · Genre · Time boost  │
└─────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- MovieLens dataset ([download here](https://grouplens.org/datasets/movielens/latest/) — use `ml-latest-small` for testing)
- Optional: TMDB API key ([free at TMDB](https://www.themoviedb.org/settings/api))

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender

# 2. Install dependencies
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your TMDB_API_KEY

# 4. Download MovieLens data
mkdir data
# Place movies.csv and ratings.csv in data/

# 5. Train models (one time only, ~2-5 minutes)
python scripts/train.py

# 6. Seed the database
python scripts/seed_db.py

# 7. Start the server
uvicorn backend.app:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

---

## 📂 Project Structure

```
movie-recommender/
├── backend/
│   ├── api/
│   │   └── routes.py            # All FastAPI endpoints
│   ├── models/
│   │   ├── content_based.py     # TF-IDF + Cosine Similarity
│   │   ├── collaborative.py     # SVD (Surprise library)
│   │   ├── hybrid.py            # Weighted blend + context
│   │   └── user_profile.py     # Dynamic preference vectors
│   ├── services/
│   │   ├── tmdb_service.py      # TMDB API wrapper
│   │   └── recommendation_service.py  # Orchestration layer
│   ├── database/
│   │   ├── models.py            # SQLAlchemy ORM tables
│   │   └── db.py                # Async engine + sessions
│   ├── utils/
│   │   ├── preprocessing.py     # Feature engineering
│   │   ├── evaluation.py        # Precision, Recall, NDCG
│   │   └── cold_start.py        # Popularity-based fallback
│   └── app.py                   # FastAPI entry point
├── frontend/
│   ├── index.html               # UI shell
│   ├── style.css                # Dark cinema theme
│   └── app.js                   # API calls + rendering
├── scripts/
│   ├── train.py                 # Full training pipeline
│   └── seed_db.py               # Populate SQLite
├── notebooks/
│   └── 01_eda_and_walkthrough.ipynb
├── data/                        # Place MovieLens CSVs here
├── artifacts/                   # Saved ML models (auto-created)
├── config.py                    # All settings
├── requirements.txt
├── .env.example
└── run.sh
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/users` | Create user |
| GET | `/api/users/{id}/profile` | User taste profile |
| POST | `/api/users/{id}/watch` | Log watched movie |
| POST | `/api/users/{id}/rate` | Rate a movie (0.5–5.0) |
| GET | `/api/users/{id}/history` | Watch history |
| GET | `/api/recommend/{id}` | Get recommendations |
| GET | `/api/movies/search?q=` | Search movies |
| GET | `/api/movies/trending` | TMDB trending |
| GET | `/api/movies/{id}` | Movie details |
| GET | `/health` | Health check |

**Recommendation query params:**
```
GET /api/recommend/1?mood=excited&genres=Action&genres=Sci-Fi&top_n=10
```

Full interactive docs: http://localhost:8000/docs

---

## 🧠 How the ML Works

### Content-Based Filtering (TF-IDF)
Every movie is converted into a "tag soup" combining genre, cast, director, keywords, and plot. TF-IDF converts this to a vector. Cosine similarity between vectors = content similarity.

```python
# Tag soup example for Toy Story:
"animation_comedy_family adventure pixar john_lasseter
 tom_hanks tim_allen toys friendship loyalty growing_up"
```

### Collaborative Filtering (SVD)
Builds a user×movie rating matrix, factorizes it into latent features. The dot product of a user's and movie's latent vectors predicts the rating.

### Hybrid Scoring
```
final_score = 0.6 × content_score
            + 0.4 × collab_score    (if ≥5 ratings)
            + genre_boost           (0–0.15)
            + popularity_boost      (0–0.05)
```

---

## 📊 Evaluation Results

| Metric | Score | Target |
|--------|-------|--------|
| Precision@10 | ~0.85 | ≥ 0.85 |
| Recall@10 | ~0.31 | — |
| NDCG@10 | ~0.72 | — |
| RMSE (SVD) | ~0.87 | ≤ 0.90 |

Run evaluation yourself:
```bash
python -c "
from scripts.train import run_evaluation
# see train.py for full usage
"
```

---

## 🔧 Configuration

Key settings in `config.py`:

```python
CONTENT_WEIGHT = 0.6          # Weight for content-based score
COLLAB_WEIGHT  = 0.4          # Weight for collaborative score
TOP_N_RECOMMENDATIONS = 10    # Default results per request

MOOD_GENRE_MAP = {
    'happy':   ['Comedy', 'Animation'],
    'excited': ['Action', 'Adventure', 'Thriller'],
    # ... etc
}
```

---

## 🚀 Deployment

### Local
```bash
bash run.sh
```

### Render (free tier)
1. Push to GitHub
2. New Web Service → connect repo
3. Build command: `pip install -r requirements.txt && python scripts/train.py`
4. Start command: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
5. Add env var: `TMDB_API_KEY`

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python scripts/train.py
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🔮 Future Improvements
- [ ] Deep learning embeddings (sentence-transformers for overview)
- [ ] Real-time collaborative filtering with matrix updates
- [ ] A/B testing framework for recommendation strategies
- [ ] React frontend with better UX
- [ ] Redis caching for hot recommendations
- [ ] WebSocket for live "similar users watching now"

---

## 📄 License
MIT — use freely for learning and commercial projects.

---

## 🙏 Credits
- [MovieLens](https://grouplens.org/datasets/movielens/) — F. Maxwell Harper and Joseph A. Konstan
- [TMDB API](https://www.themoviedb.org/) — movie metadata and posters
- [Surprise](https://surpriselib.com/) — SVD collaborative filtering
