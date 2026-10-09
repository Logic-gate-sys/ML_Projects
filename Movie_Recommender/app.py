"""
Movie ML Lab - Main Application
Interactive Streamlit dashboard for exploring movie recommendation ML models.
"""

import streamlit as st
from src.config import settings

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=settings.APP_NAME,
    page_icon=settings.APP_ICON,
    layout=settings.LAYOUT,
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        height: 100%;
        cursor: pointer;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #8b5cf6;
    }
    
    /* Navigation grid */
    .nav-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    /* Quick links */
    .quick-link {
        background: #f9fafb;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .quick-link:hover {
        background: #f3f4f6;
        transform: translateX(5px);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HERO SECTION
# ============================================================================

st.markdown("""
<div class="hero-section">
    <h1 style="margin: 0; font-size: 3.5rem; font-weight: 800;">
        🎬 Movie ML Lab
    </h1>
    <p style="font-size: 1.5rem; margin: 1rem 0 0 0; opacity: 0.95; font-weight: 500;">
        Interactive Machine Learning Dashboard for Movie Recommendations
    </p>
    <p style="font-size: 1.1rem; margin: 1rem 0 0 0; opacity: 0.9;">
        Explore 85K+ movies • 5 ML models • 175 features • Real-time predictions
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# KEY METRICS
# ============================================================================

st.markdown("## 📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-card">
        <h2 style="margin: 0; color: #8b5cf6; font-size: 2.5rem;">85K+</h2>
        <p style="margin: 0.5rem 0 0 0; color: #6b7280; font-weight: 600;">Movies</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <h2 style="margin: 0; color: #8b5cf6; font-size: 2.5rem;">33M+</h2>
        <p style="margin: 0.5rem 0 0 0; color: #6b7280; font-weight: 600;">Ratings</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <h2 style="margin: 0; color: #8b5cf6; font-size: 2.5rem;">175</h2>
        <p style="margin: 0.5rem 0 0 0; color: #6b7280; font-weight: 600;">Features</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <h2 style="margin: 0; color: #8b5cf6; font-size: 2.5rem;">5</h2>
        <p style="margin: 0.5rem 0 0 0; color: #6b7280; font-weight: 600;">ML Models</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# NAVIGATION CARDS
# ============================================================================

st.markdown("---")
st.markdown("## Explore the Dashboard")

# Create 2x3 grid of feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🏠</div>
        <h3 style="color: #1f2937; margin: 0 0 0.5rem 0;">Overview</h3>
        <p style="color: #6b7280; margin: 0; line-height: 1.6;">
            Interactive dashboard with key metrics, visualizations, and model performance insights.
        </p>
        <br>
        <ul style="color: #6b7280; margin: 0; padding-left: 1.2rem;">
            <li>Dataset statistics</li>
            <li>Model comparisons</li>
            <li>Feature engineering</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
        <h3 style="color: #1f2937; margin: 0 0 0.5rem 0;">Data Explorer</h3>
        <p style="color: #6b7280; margin: 0; line-height: 1.6;">
            Browse, search, and analyze 85,000+ movies with advanced filtering options.
        </p>
        <br>
        <ul style="color: #6b7280; margin: 0; padding-left: 1.2rem;">
            <li>Search by title/genre</li>
            <li>Filter by rating</li>
            <li>Sort and export data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
        <h3 style="color: #1f2937; margin: 0 0 0.5rem 0;">Model Comparison</h3>
        <p style="color: #6b7280; margin: 0; line-height: 1.6;">
            Compare 5 ML models: XGBoost, LightGBM, Ridge, Logistic, RandomForest.
        </p>
        <br>
        <ul style="color: #6b7280; margin: 0; padding-left: 1.2rem;">
            <li>Performance metrics</li>
            <li>Visual comparisons</li>
            <li>Model strengths</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🎬</div>
        <h3 style="color: #1f2937; margin: 0 0 0.5rem 0;">Content Recommender</h3>
        <p style="color: #6b7280; margin: 0; line-height: 1.6;">
            Get recommendations based on movie features: genres, tags, and temporal data.
        </p>
        <br>
        <ul style="color: #6b7280; margin: 0; padding-left: 1.2rem;">
            <li>Genre-based matching</li>
            <li>Content similarity</li>
            <li>Instant results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size: 3rem; margin-bottom: 1rem;">👥</div>
        <h3 style="color: #1f2937; margin: 0 0 0.5rem 0;">Collaborative Filtering</h3>
        <p style="color: #6b7280; margin: 0; line-height: 1.6;">
            Discover movies based on user behavior patterns and rating similarities.
        </p>
        <br>
        <ul style="color: #6b7280; margin: 0; padding-left: 1.2rem;">
            <li>User pattern analysis</li>
            <li>Behavior-based</li>
            <li>Multiple CF methods</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card" style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-color: #f59e0b;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
        <h3 style="color: #1f2937; margin: 0 0 0.5rem 0;">Quick Start</h3>
        <p style="color: #6b7280; margin: 0; line-height: 1.6;">
            New here? Start with Overview for a complete introduction to the system.
        </p>
        <br>
        <ul style="color: #6b7280; margin: 0; padding-left: 1.2rem;">
            <li>Guided walkthrough</li>
            <li>Feature explanations</li>
            <li>Best practices</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PROJECT DETAILS
# ============================================================================

st.markdown("---")
st.markdown("## 🔬 Technical Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📚 Dataset: MovieLens 25M
    
    - **Source:** GroupLens Research
    - **Size:** 33 million ratings
    - **Movies:** 85,555 titles
    - **Users:** 261,000+ users
    - **Time span:** 1995-2019
    - **Genres:** 20 categories
    - **Tags:** 1,128 genome tags
    
    ### Machine Learning Task
    
    **Objective:** Predict user ratings for movies  
    **Type:** Regression problem  
    **Target:** Rating (0.5 - 5.0 stars)  
    **Metric:** RMSE (Root Mean Squared Error)
    """)

with col2:
    st.markdown("""
    ### 🧮 Feature Engineering (175 features)
    
    - **29 Genre features** - One-hot encoded genres
    - **50 Title features** - TF-IDF vectors (SVD reduced)
    - **75 Genome features** - Tag relevance scores
    - **21 Temporal features** - Release year, age, era
    
    ### 🤖 ML Models (5 algorithms)
    
    1. **XGBoost** - Gradient boosting (RMSE: 0.89)
    2. **LightGBM** - Fast gradient boosting (RMSE: 0.90)
    3. **Ridge** - Regularized linear regression (RMSE: 0.94)
    4. **Logistic** - Classification-based (RMSE: 1.02)
    5. **RandomForest** - Ensemble trees (RMSE: 0.95)
    """)

# ============================================================================
# QUICK LINKS
# ============================================================================

st.markdown("---")
st.markdown("## Quick Actions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="quick-link">
        <strong>🎬 Try Content Recommender</strong><br>
        <small>Get instant movie recommendations based on genres</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="quick-link">
        <strong>📊 Compare ML Models</strong><br>
        <small>See which algorithm performs best</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="quick-link">
        <strong>🔍 Explore Dataset</strong><br>
        <small>Browse 85,000+ movies with filters</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="quick-link">
        <strong>👥 Collaborative Filtering</strong><br>
        <small>Discover movies based on user behavior</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="quick-link">
        <strong>🏠 View Dashboard</strong><br>
        <small>Interactive overview with visualizations</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="quick-link">
        <strong>📈 Feature Engineering</strong><br>
        <small>Learn how 175 features were created</small>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# TIPS & FOOTER
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("💡 **Tip:** All data is cached for instant loading. Restart app if you update source files.")

with col2:
    st.success("✨ **Best Feature:** Compare all 5 ML models side-by-side with interactive charts!")

with col3:
    st.warning(" **Recommended:** Start with Overview page for complete feature walkthrough.")

# ============================================================================
# ABOUT SECTION
# ============================================================================

st.markdown("---")

with st.expander("ℹ️ About This Project"):
    st.markdown("""
    ### 🎓 Movie ML Lab - Interactive Recommendation System
    
    This Streamlit dashboard demonstrates a complete machine learning pipeline for movie recommendations:
    
    #### 🔄 Pipeline Stages:
    
    1. **Data Collection** - MovieLens 25M dataset
    2. **Feature Engineering** - 175 content features
    3. **Model Training** - 5 ML algorithms
    4. **Evaluation** - RMSE, MAE, R² metrics
    5. **Deployment** - Interactive web app
    
    #### Key Features:
    
    - **Content-Based Filtering** - Genre and tag similarity
    - **Collaborative Filtering** - User behavior patterns
    - **Model Comparison** - Side-by-side performance
    - **Data Exploration** - Advanced search and filtering
    - **Real-Time Predictions** - Instant recommendations
    
    #### 📊 Technologies Used:
    
    - **Framework:** Streamlit
    - **ML Libraries:** Scikit-learn, XGBoost, LightGBM
    - **Data:** Pandas, NumPy
    - **Visualization:** Plotly, Matplotlib
    
    #### 👨‍💻 Developed By:
    
    Machine Learning enthusiasts passionate about recommendation systems and interactive data science.
    
    ---
    
    **Version:** 1.0.0 | **Last Updated:** December 2025
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f9fafb; border-radius: 12px;">
    <h3 style="color: #1f2937; margin: 0 0 0.5rem 0;">Ready to Explore?</h3>
    <p style="color: #6b7280; margin: 0;">
        Navigate using the sidebar on the left to start your journey through the Movie ML Lab!
    </p>
    <br>
    <p style="color: #9ca3af; font-size: 0.9rem; margin: 0;">
        🎬 Movie ML Lab • Built with Streamlit • Powered by MovieLens Data
    </p>
</div>
""", unsafe_allow_html=True)
