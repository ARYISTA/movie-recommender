"""
database/models.py — SQLAlchemy ORM table definitions
Each class = one database table.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    ForeignKey, Text, Boolean, JSON
)
from sqlalchemy.orm import relationship, DeclarativeBase


class Base(DeclarativeBase):
    """All ORM models inherit from this base class."""
    pass


class User(Base):
    """
    Represents a registered user.
    'preference_vector' stores a JSON dict of genre → score
    so we can quickly retrieve what genres they like most.
    """
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(80), unique=True, nullable=False, index=True)
    email         = Column(String(120), unique=True, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    last_active   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Stored as JSON: {"Action": 0.8, "Comedy": 0.4, ...}
    preference_vector = Column(JSON, default=dict)

    # Relationships
    history  = relationship("WatchHistory", back_populates="user", cascade="all, delete")
    ratings  = relationship("Rating",       back_populates="user", cascade="all, delete")

    def __repr__(self):
        return f"<User {self.username}>"


class Movie(Base):
    """
    Core movie table populated from MovieLens + enriched via TMDB.
    'tag_soup' is the pre-processed text we feed into TF-IDF:
    a combination of genres, keywords, cast, director, and overview.
    """
    __tablename__ = "movies"

    id           = Column(Integer, primary_key=True, index=True)  # MovieLens movieId
    title        = Column(String(255), nullable=False, index=True)
    genres       = Column(String(500), nullable=True)   # pipe-separated: "Action|Comedy"
    year         = Column(Integer, nullable=True)
    overview     = Column(Text, nullable=True)
    cast         = Column(String(500), nullable=True)   # top-3 actors, pipe-separated
    director     = Column(String(200), nullable=True)
    keywords     = Column(String(500), nullable=True)
    tmdb_id      = Column(Integer, nullable=True, index=True)
    poster_path  = Column(String(300), nullable=True)
    vote_average = Column(Float, default=0.0)
    popularity   = Column(Float, default=0.0)
    tag_soup     = Column(Text, nullable=True)   # pre-built feature string for TF-IDF

    # Relationships
    history  = relationship("WatchHistory", back_populates="movie")
    ratings  = relationship("Rating",       back_populates="movie")

    def genres_list(self) -> list[str]:
        """Return genres as a Python list."""
        return self.genres.split("|") if self.genres else []

    def __repr__(self):
        return f"<Movie {self.title} ({self.year})>"


class WatchHistory(Base):
    """
    Tracks every movie a user has watched.
    'watch_progress' (0–100) lets us know if they finished it.
    """
    __tablename__ = "watch_history"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    movie_id      = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
    watched_at    = Column(DateTime, default=datetime.utcnow, index=True)
    watch_progress = Column(Integer, default=100)  # 0-100 percent

    user  = relationship("User",  back_populates="history")
    movie = relationship("Movie", back_populates="history")


class Rating(Base):
    """Explicit star rating (0.5–5.0) from a user for a movie."""
    __tablename__ = "ratings"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id",  ondelete="CASCADE"), nullable=False)
    movie_id   = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
    score      = Column(Float, nullable=False)          # 0.5 – 5.0
    rated_at   = Column(DateTime, default=datetime.utcnow)

    user  = relationship("User",  back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")
