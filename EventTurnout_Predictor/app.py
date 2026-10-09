import streamlit as st
import pandas as pd
import joblib

# Load the model pipelines (linear regression and logistic regression)
def load_model(model_type):
    if model_type == "Linear Regression":
        return joblib.load('models/lr_pipeline.joblib')
    elif model_type == "Logistic Regression":
        return joblib.load('models\log_pipeline.joblib')

# Load datasets
linear_regression_data = pd.read_csv('models/university_event_turnout_data.csv')
logistic_regression_data = pd.read_csv('models\student_event_attendance.csv')
#drop student id column from logistic regression data
logistic_regression_data.drop(columns=['student_id'], inplace=True)  # Drop student_id column
# Streamlit page setup
st.set_page_config(page_title="Event Predictor", page_icon="📈", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
            font-family: 'Poppins', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            color: #333333;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 24px;
            color: #666666;
            margin-bottom: 30px;
        }
        .input-container {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0px 8px 16px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .prediction-container {
            background-color: #e8f5e9;
            padding: 25px;
            border-radius: 12px;
            font-size: 26px;
            font-weight: bold;
            color: #2e7d32;
            text-align: center;
            box-shadow: 0px 8px 16px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="title">📈 Event Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Choose a model and fill in the details to predict attendance or likelihood of attendance.</div>', unsafe_allow_html=True)

# Model Selection
st.sidebar.header("Select Model")
model_type = st.sidebar.selectbox("Choose a model:", ["Linear Regression", "Logistic Regression"])

# Select the appropriate dataset based on the model
if model_type == "Linear Regression":
    dataset = linear_regression_data
    target_description = "Predicted Attendance"
else:
    dataset = logistic_regression_data
    target_description = "Likelihood of Attendance"

feature_columns = dataset.columns[:-1]  # All columns except the target column

# Input Section
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.subheader(f"🔹 Enter Information for {model_type}")

user_input = {}
for col in feature_columns:
    if dataset[col].dtype == 'object':
        unique_vals = sorted(dataset[col].dropna().unique())
        user_input[col] = st.selectbox(f"{col}:", unique_vals)
    elif dataset[col].dtype == 'int64':
        user_input[col] = st.number_input(
            f"{col}:",
            min_value=int(dataset[col].min()),
            max_value=int(dataset[col].max()),
            value=int(dataset[col].mean()),
            step=1  
        )
    elif dataset[col].dtype == 'float64':
        user_input[col] = st.number_input(
            f"{col}:",
            min_value=float(dataset[col].min()),
            max_value=float(dataset[col].max()),
            value=float(dataset[col].mean()),
            step=1.0  # Ensure step is a float
        )
    else:
        st.warning(f" Unsupported column type for {col}")
st.markdown('</div>', unsafe_allow_html=True)

# Prepare input for model
input_df = pd.DataFrame([user_input])

# Load selected model pipeline
model = load_model(model_type)

# Prediction
if st.button("Predict"):
    with st.spinner('Predicting... ⏳'):
        try:
            # Use the selected model pipeline to preprocess and predict
            prediction = model.predict(input_df)
            if model_type == "Linear Regression":
                st.markdown(f'<div class="prediction-container">🎯 {target_description}: {prediction[0]:,.0f} people</div>', unsafe_allow_html=True)
            elif model_type == "Logistic Regression":
                probability = model.predict_proba(input_df)[0][1]  # Probability of class 1
                st.markdown(f'<div class="prediction-container">🎯 {target_description}: {probability * 100:.2f}%</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")