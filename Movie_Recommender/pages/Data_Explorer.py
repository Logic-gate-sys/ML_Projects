"""Data Explorer page."""
import streamlit as st
import pandas as pd
from src.config import settings
from src.data.loaders import load_movies, load_movie_stats

st.set_page_config(
    page_title=f"{settings.APP_NAME} - Data Explorer",
    page_icon=settings.APP_ICON,
    layout=settings.LAYOUT
)

st.title("🔍 Data Explorer")
st.markdown("**Browse and search through 85,000+ movies**")
st.markdown("---")

# Load data
with st.spinner("📊 Loading movie data..."):
    movies = load_movies()
    movie_stats = load_movie_stats()
    
    if not movie_stats.empty and 'movieId' in movies.columns:
        df = movies.merge(movie_stats, on='movieId', how='left')
    else:
        df = movies.copy()

# Search and filters
st.markdown("## 🔎 Search & Filter")

col1, col2 = st.columns([3, 1])

with col1:
    search = st.text_input(
        "Search movies by title",
        placeholder="Try: Matrix, Inception, Star Wars...",
        help="Search for movies by title"
    )

with col2:
    search_type = st.selectbox(
        "Search in",
        ["Title", "Genre"],
        help="Choose what to search"
    )

# Advanced filters
with st.expander("🎛️ Advanced Filters"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'avg_rating_train' in df.columns:
            min_rating = st.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.5)
        else:
            min_rating = 0.0
    
    with col2:
        if 'num_ratings_train' in df.columns:
            min_votes = st.number_input("Minimum Votes", 0, 10000, 0, 100)
        else:
            min_votes = 0
    
    with col3:
        if 'genres_list' in df.columns:
            all_genres = set()
            for genres in df['genres_list'].dropna():
                if isinstance(genres, list):
                    all_genres.update(genres)
            
            filter_genres = st.multiselect("Filter by Genre", sorted(all_genres))
        else:
            filter_genres = []

# Apply filters
filtered_df = df.copy()

if search:
    if search_type == "Title":
        filtered_df = filtered_df[filtered_df['title'].str.contains(search, case=False, na=False)]
    else:
        if 'genres' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['genres'].str.contains(search, case=False, na=False)]

if 'avg_rating_train' in filtered_df.columns and min_rating > 0:
    filtered_df = filtered_df[filtered_df['avg_rating_train'] >= min_rating]

if 'num_ratings_train' in filtered_df.columns and min_votes > 0:
    filtered_df = filtered_df[filtered_df['num_ratings_train'] >= min_votes]

if filter_genres and 'genres_list' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['genres_list'].apply(
        lambda x: any(g in x for g in filter_genres) if isinstance(x, list) else False
    )]

# Display results
st.markdown("---")
st.markdown(f"## 📊 Results ({len(filtered_df):,} movies)")

if len(filtered_df) > 0:
    # Sort options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        sort_options = ['title']
        if 'avg_rating_train' in filtered_df.columns:
            sort_options.append('avg_rating_train')
        if 'num_ratings_train' in filtered_df.columns:
            sort_options.append('num_ratings_train')
        
        sort_by = st.selectbox("Sort by", sort_options)
    
    with col2:
        sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)
    
    # Sort data
    ascending = (sort_order == "Ascending")
    display_df = filtered_df.sort_values(sort_by, ascending=ascending)
    
    # Display columns
    display_cols = ['title', 'genres']
    if 'avg_rating_train' in display_df.columns:
        display_cols.append('avg_rating_train')
    if 'num_ratings_train' in display_df.columns:
        display_cols.append('num_ratings_train')
    
    # Show data
    st.dataframe(
        display_df[display_cols].head(100),
        use_container_width=True,
        hide_index=True
    )
    
    if len(display_df) > 100:
        st.info(f"Showing first 100 of {len(display_df):,} results")
    
    # Export
    st.markdown("---")
    csv = display_df[display_cols].to_csv(index=False)
    st.download_button(
        "📥 Export Results to CSV",
        data=csv,
        file_name="movie_explorer_results.csv",
        mime="text/csv"
    )
else:
    st.warning("No movies found matching your criteria.")
