"""
Data loading module.
"""

from src.data.loaders import (
    load_movies,
    load_movie_stats,
    load_ratings,
    load_genome_scores,
    get_dataset_stats
)

__all__ = [
    "load_movies",
    "load_movie_stats",
    "load_ratings",
    "load_genome_scores",
    "get_dataset_stats"
]
