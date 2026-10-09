"""
Configuration settings for Movie ML Lab.
"""

class Settings:
    """Application settings."""
    
    # App metadata
    APP_NAME = "Movie ML Lab"
    APP_ICON = "🎬"
    LAYOUT = "wide"
    
    # Data paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    
    # Feature counts
    N_GENRE_FEATURES = 29
    N_TITLE_FEATURES = 50
    N_GENOME_FEATURES = 75
    N_TEMPORAL_FEATURES = 21
    TOTAL_FEATURES = 175
    
    # Model names
    MODELS = [
        "XGBoost",
        "LightGBM",
        "Ridge",
        "Logistic Regression",
        "Random Forest"
    ]
    
    # Color scheme
    PRIMARY_COLOR = "#667eea"
    SECONDARY_COLOR = "#764ba2"
    
    # Cache TTL (seconds)
    CACHE_TTL = 3600

# Create settings instance
settings = Settings()
