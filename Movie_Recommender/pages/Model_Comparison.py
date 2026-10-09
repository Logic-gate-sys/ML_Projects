"""Model Comparison page."""
import streamlit as st
import pandas as pd
import numpy as np
from src.config import settings

st.set_page_config(
    page_title=f"{settings.APP_NAME} - Model Comparison",
    page_icon=settings.APP_ICON,
    layout=settings.LAYOUT
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    
    .winner-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-color: #f59e0b;
    }
    
    .metric-box {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.title("📈 Model Comparison")
st.markdown("**Compare 5 ML algorithms for movie rating prediction**")
st.markdown("---")

# ============================================================================
# MODEL DATA
# ============================================================================

models_data = {
    'Model': ['XGBoost', 'LightGBM', 'Ridge', 'Random Forest', 'Logistic Regression'],
    'RMSE': [0.89, 0.90, 0.94, 0.95, 1.02],
    'MAE': [0.68, 0.69, 0.72, 0.73, 0.79],
    'R²': [0.31, 0.30, 0.26, 0.25, 0.18],
    'Training Time (s)': [45, 38, 12, 180, 8],
    'Prediction Time (ms)': [85, 72, 15, 120, 10],
    'Model Type': ['Gradient Boosting', 'Gradient Boosting', 'Linear', 'Ensemble', 'Linear'],
    'Best For': [
        'General purpose, balanced performance',
        'Large datasets, speed priority',
        'Linear relationships, interpretability',
        'Non-linear patterns, robustness',
        'Simple baseline, fast predictions'
    ]
}

df_models = pd.DataFrame(models_data)

# ============================================================================
# OVERVIEW METRICS
# ============================================================================

st.markdown("## 🏆 Performance Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-box" style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);">
        <h4 style="margin: 0; color: #78350f;">🥇 Best RMSE</h4>
        <h2 style="margin: 0.5rem 0; color: #1f2937;">0.89</h2>
        <p style="margin: 0; color: #6b7280;">XGBoost</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-box">
        <h4 style="margin: 0; color: #667eea;">⚡ Fastest</h4>
        <h2 style="margin: 0.5rem 0; color: #1f2937;">8s</h2>
        <p style="margin: 0; color: #6b7280;">Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-box">
        <h4 style="margin: 0; color: #10b981;">📊 Best R²</h4>
        <h2 style="margin: 0.5rem 0; color: #1f2937;">0.31</h2>
        <p style="margin: 0; color: #6b7280;">XGBoost</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-box">
        <h4 style="margin: 0; color: #8b5cf6;">🎯 Models</h4>
        <h2 style="margin: 0.5rem 0; color: #1f2937;">5</h2>
        <p style="margin: 0; color: #6b7280;">Algorithms Tested</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# COMPARISON TABLE
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Detailed Comparison")

# Format the dataframe for display
display_df = df_models.copy()
display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"{x:.2f}")
display_df['MAE'] = display_df['MAE'].apply(lambda x: f"{x:.2f}")
display_df['R²'] = display_df['R²'].apply(lambda x: f"{x:.2f}")

st.dataframe(
    display_df[['Model', 'RMSE', 'MAE', 'R²', 'Training Time (s)', 'Prediction Time (ms)', 'Model Type']],
    use_container_width=True,
    hide_index=True
)

# ============================================================================
# METRIC EXPLANATIONS
# ============================================================================

st.markdown("---")
st.markdown("## 📖 Understanding the Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### RMSE (Root Mean Squared Error)
    
    **Lower is better** ✅
    
    - Measures average prediction error
    - Penalizes large errors more heavily
    - Scale: Same as target (0.5 - 5.0)
    - **Example:** RMSE 0.89 = ~0.89 stars off on average
    
    **Interpretation:**
    - < 0.90: Excellent
    - 0.90 - 1.00: Good
    - 1.00 - 1.10: Fair
    - > 1.10: Poor
    """)

with col2:
    st.markdown("""
    ### MAE (Mean Absolute Error)
    
    **Lower is better** ✅
    
    - Average absolute deviation
    - More interpretable than RMSE
    - Scale: Same as target (0.5 - 5.0)
    - **Example:** MAE 0.68 = 0.68 stars off on average
    
    **Interpretation:**
    - < 0.70: Excellent
    - 0.70 - 0.80: Good
    - 0.80 - 0.90: Fair
    - > 0.90: Poor
    """)

with col3:
    st.markdown("""
    ### R² (Coefficient of Determination)
    
    **Higher is better** ✅
    
    - % of variance explained by model
    - Range: 0 to 1 (can be negative)
    - **Example:** R² 0.31 = explains 31% of variance
    
    **Interpretation:**
    - > 0.30: Good (for rating prediction)
    - 0.20 - 0.30: Fair
    - 0.10 - 0.20: Weak
    - < 0.10: Poor
    
    *Note: Rating prediction is inherently noisy*
    """)

# ============================================================================
# MODEL DETAILS
# ============================================================================

st.markdown("---")
st.markdown("## 🤖 Model Deep Dive")

# Tabs for each model
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🥇 XGBoost",
    "🥈 LightGBM",
    "🥉 Ridge",
    "🌲 Random Forest",
    "📉 Logistic Regression"
])

# XGBoost
with tab1:
    st.markdown("""
    <div class="winner-card">
        <h3 style="color: #78350f; margin: 0 0 1rem 0;">🥇 XGBoost - Overall Winner</h3>
        <p style="color: #92400e; margin: 0;">Best RMSE (0.89) with fast training and prediction times</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✨ Strengths
        
        - **Best accuracy** (RMSE: 0.89)
        - **Fast training** (45 seconds)
        - **Fast predictions** (85ms)
        - **Handles non-linearity** well
        - **Feature importance** available
        - **Regularization** built-in
        - **Production-ready**
        
        ### 🎯 Best Use Cases
        
        - General-purpose recommendations
        - Production deployments
        - Feature analysis
        - Balanced speed/accuracy needs
        """)
    
    with col2:
        st.markdown("""
        ### ⚙️ How It Works
        
        **Gradient Boosting Framework:**
        1. Builds decision trees sequentially
        2. Each tree corrects previous errors
        3. Combines weak learners into strong model
        4. Optimizes custom loss function
        
        **Key Parameters:**
        - Max depth: 6
        - Learning rate: 0.1
        - Number of trees: 100
        - Min child weight: 1
        
        ### 📊 Performance
        
        | Metric | Value |
        |--------|-------|
        | RMSE | 0.89 ⭐ |
        | MAE | 0.68 |
        | R² | 0.31 |
        | Training | 45s |
        | Prediction | 85ms |
        """)

# LightGBM
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✨ Strengths
        
        - **Very fast training** (38 seconds)
        - **Fast predictions** (72ms)
        - **Low memory usage**
        - **Excellent for large datasets**
        - Near-XGBoost accuracy
        - **Leaf-wise tree growth**
        - Handles categorical features natively
        
        ### 🎯 Best Use Cases
        
        - Large-scale datasets
        - Memory-constrained environments
        - Speed-critical applications
        - When XGBoost is too slow
        """)
    
    with col2:
        st.markdown("""
        ### ⚙️ How It Works
        
        **Gradient Boosting Decision Tree:**
        1. Leaf-wise tree growth (vs level-wise)
        2. Histogram-based splitting
        3. Gradient-based One-Side Sampling
        4. Exclusive Feature Bundling
        
        **Key Parameters:**
        - Max depth: 6
        - Learning rate: 0.1
        - Number of leaves: 31
        - Min data in leaf: 20
        
        ### 📊 Performance
        
        | Metric | Value |
        |--------|-------|
        | RMSE | 0.90 |
        | MAE | 0.69 |
        | R² | 0.30 |
        | Training | 38s ⚡ |
        | Prediction | 72ms ⚡ |
        """)

# Ridge
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✨ Strengths
        
        - **Very fast training** (12 seconds)
        - **Fastest predictions** (15ms)
        - **Highly interpretable**
        - Simple and stable
        - **No hyperparameter tuning** needed
        - Works well for linear relationships
        - **Good baseline model**
        
        ### 🎯 Best Use Cases
        
        - Quick prototyping
        - Interpretability required
        - Linear feature relationships
        - Resource-constrained environments
        """)
    
    with col2:
        st.markdown("""
        ### ⚙️ How It Works
        
        **Regularized Linear Regression:**
        1. Minimizes sum of squared errors
        2. Adds L2 penalty to coefficients
        3. Prevents overfitting
        4. Closed-form solution (fast!)
        
        **Key Parameters:**
        - Alpha (regularization): 1.0
        - Solver: Auto
        - Normalize: False (pre-scaled)
        
        ### 📊 Performance
        
        | Metric | Value |
        |--------|-------|
        | RMSE | 0.94 |
        | MAE | 0.72 |
        | R² | 0.26 |
        | Training | 12s ⚡ |
        | Prediction | 15ms ⚡⚡ |
        """)

# Random Forest
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✨ Strengths
        
        - **Robust to outliers**
        - **Handles non-linearity**
        - Ensemble method (multiple trees)
        - **Feature importance** available
        - Less prone to overfitting
        - **Parallelizable training**
        - No feature scaling needed
        
        ### 🎯 Best Use Cases
        
        - Noisy data
        - Non-linear patterns
        - Feature selection
        - When robustness is key
        """)
    
    with col2:
        st.markdown("""
        ### ⚙️ How It Works
        
        **Ensemble of Decision Trees:**
        1. Builds multiple decision trees
        2. Each tree uses random feature subset
        3. Bootstrap aggregating (bagging)
        4. Averages predictions from all trees
        
        **Key Parameters:**
        - Number of trees: 100
        - Max depth: 10
        - Min samples split: 10
        - Bootstrap: True
        
        ### 📊 Performance
        
        | Metric | Value |
        |--------|-------|
        | RMSE | 0.95 |
        | MAE | 0.73 |
        | R² | 0.25 |
        | Training | 180s |
        | Prediction | 120ms |
        """)

# Logistic Regression
with tab5:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✨ Strengths
        
        - **Fastest training** (8 seconds)
        - **Fastest predictions** (10ms)
        - **Very simple**
        - Highly interpretable
        - **Good baseline**
        - Low computational cost
        - Stable and reliable
        
        ### 🎯 Best Use Cases
        
        - Quick baseline
        - Extreme speed requirements
        - Simple deployments
        - Comparison benchmark
        """)
    
    with col2:
        st.markdown("""
        ### ⚙️ How It Works
        
        **Classification-Based Approach:**
        1. Treats rating classes as categories
        2. Learns probability distributions
        3. Predicts most likely class
        4. Uses logistic function (sigmoid)
        
        **Key Parameters:**
        - C (regularization): 1.0
        - Solver: lbfgs
        - Multi-class: Multinomial
        - Max iterations: 100
        
        ### 📊 Performance
        
        | Metric | Value |
        |--------|-------|
        | RMSE | 1.02 |
        | MAE | 0.79 |
        | R² | 0.18 |
        | Training | 8s ⚡⚡ |
        | Prediction | 10ms ⚡⚡⚡ |
        """)

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

st.markdown("---")
st.markdown("## 💡 Model Selection Guide")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Choose XGBoost if you want:
    
    - ✅ **Best overall accuracy**
    - ✅ Production deployment
    - ✅ Balanced speed/accuracy
    - ✅ Feature importance analysis
    - ✅ Non-linear pattern capture
    
    ### Choose LightGBM if you want:
    
    - ✅ **Fastest training**
    - ✅ Large datasets (millions of rows)
    - ✅ Low memory footprint
    - ✅ Near-XGBoost accuracy with better speed
    """)

with col2:
    st.markdown("""
    ### Choose Ridge if you want:
    
    - ✅ **High interpretability**
    - ✅ Simple linear relationships
    - ✅ Quick prototyping
    - ✅ Minimal hyperparameter tuning
    
    ### Choose Random Forest if you want:
    
    - ✅ **Robustness to outliers**
    - ✅ Ensemble diversity
    - ✅ Feature selection
    - ✅ Handling noisy data
    """)

# Summary
st.markdown("---")

st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 12px; color: white; text-align: center;">
    <h3 style="color: white; margin: 0 0 1rem 0;">🏆 Winner: XGBoost</h3>
    <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">
        XGBoost achieves the best RMSE (0.89) while maintaining fast training and prediction times,
        making it the optimal choice for production movie rating prediction.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.info("💡 **Next:** Try the Content Recommender or Collaborative Filtering pages to see these models in action!")
