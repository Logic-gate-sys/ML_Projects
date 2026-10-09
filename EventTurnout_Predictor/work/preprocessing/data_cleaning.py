import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess(file_path, model_type='regression'):
    """
    Load dataset, preprocess features professionally for regression or classification.
    model_type: 'regression' for Linear Regression, 'classification' for Logistic Regression
    """
    # Load data
    df = pd.read_csv(file_path)
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)  # Use median for numeric
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)  # Use mode for categorical
    
    # Optional: Handle obvious outliers
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    # --- Step 2: Feature-Target Separation ---
    if model_type == 'regression':
        target = 'Actual_Attendance'  
    elif model_type == 'classification':
        target = 'attended' 
        df.drop(columns=['student_id'], inplace=True)  
    else:
        raise ValueError("model_type must be 'regression' or 'classification'")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    # --- Step 3: Identify Column Types ---
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # --- Step 4: Build Preprocessing Pipelines ---
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))  # Avoid dummy variable trap
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    
    # --- Step 5: Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
