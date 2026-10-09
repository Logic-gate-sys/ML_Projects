"""
Data loading functions with caching for Movie ML Lab.
"""

import streamlit as st
import pandas as pd
import os

@st.cache_data(ttl=3600, show_spinner=False)
def load_movies():
    """
    Load movies dataset with caching.
    
    Returns:
        pd.DataFrame: Movies data with columns: movieId, title, genres, year
    """
    try:
        df = pd.read_csv("data/movies_cleaned.csv")
        
        # Parse genres into list
        if 'genres' in df.columns:
            df['genres_list'] = df['genres'].apply(
                lambda x: x.split('|') if isinstance(x, str) and x != '(no genres listed)' else []
            )
        
        return df
    except FileNotFoundError:
        st.error("❌ movies_cleaned.csv not found in data/ directory")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_movie_stats():
    """
    Load movie statistics (ratings aggregates).
    
    Returns:
        pd.DataFrame: Movie stats with avg_rating, num_ratings, etc.
    """
    try:
        df = pd.read_csv("data/baseline_movie_stats.csv")
        return df
    except FileNotFoundError:
        st.warning("⚠️ baseline_movie_stats.csv not found - using movies only")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_ratings():
    """
    Load ratings dataset (optional - large file).
    
    Returns:
        pd.DataFrame: Ratings data
    """
    try:
        df = pd.read_csv("data/ratings_cleaned.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_genome_scores():
    """
    Load genome tag scores (optional - large file).
    
    Returns:
        pd.DataFrame: Genome scores
    """
    try:
        df = pd.read_csv("data/genome_scores_cleaned.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def get_dataset_stats():
    """
    Get overall dataset statistics.
    
    Returns:
        dict: Statistics dictionary
    """
    movies = load_movies()
    stats = load_movie_stats()
    
    return {
        'num_movies': len(movies),
        'num_ratings': stats['num_ratings_train'].sum() if 'num_ratings_train' in stats.columns else 0,
        'avg_rating': stats['avg_rating_train'].mean() if 'avg_rating_train' in stats.columns else 0,
        'num_genres': len(set([g for genres in movies['genres_list'] for g in genres])) if 'genres_list' in movies.columns else 0
    }
