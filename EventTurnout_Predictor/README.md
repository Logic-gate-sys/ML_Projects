Event Predictor Application
Overview
The Event Predictor is a Streamlit-based web application designed to predict:

Event Attendance using a Linear Regression model.
Likelihood of Attendance using a Logistic Regression model.
The app allows users to input relevant features, select a prediction model, and view the results in an intuitive and visually appealing interface.

Features
Model Selection: Choose between Linear Regression and Logistic Regression models.
Dynamic Input Forms: Automatically generate input fields based on the selected model's dataset.
Custom Styling: A clean and modern UI with custom CSS for better user experience.
Real-time Predictions: Get predictions instantly after submitting inputs.

How It Works
Model Selection:

Users select either "Linear Regression" or "Logistic Regression" from the sidebar.
The app dynamically loads the appropriate dataset and model.
Input Features:

The app generates input fields for all features in the dataset (excluding the target column).
Users can input numeric values or select categorical options based on the dataset's structure.
Prediction:

For Linear Regression: Predicts the number of attendees for an event.
For Logistic Regression: Predicts the probability of attendance as a percentage.
Results:

The prediction is displayed in a styled container with clear and concise output.

Installation and Setup
Prerequisites
Python 3.8 or higher
Required Python libraries:
streamlit
pandas
joblib
scikit-learn


Steps
Clone the repository or download the project files.
Navigate to the project directory:
Install the required dependencies:
Run the Streamlit app:
Open the app in your browser:
Local URL: http://localhost:8501 or clone the repository and run "streamlit run app.py" 
Network URL: Provided in the terminal output.


Project/
│
├── app.py                     # Main Streamlit application
├── models/
│   ├── lr_pipeline.joblib     # Linear Regression model
│   ├── log_pipeline.joblib    # Logistic Regression model
│
    ├── data/
│   ├── university_event_turnout_data.csv  # Dataset for Linear Regression
│   ├── student_event_attendance.csv       # Dataset for Logistic Regression
│
└── requirements.txt           # Python dependencies


Usage Instructions
1. Select a Model
Use the sidebar to choose between:
Linear Regression: Predicts the number of attendees.
Logistic Regression: Predicts the likelihood of attendance.
2. Input Features
Fill in the dynamically generated input fields based on the selected model's dataset:
Numeric Fields: Enter values within the range of the dataset.
Categorical Fields: Select from dropdown options.
3. Get Predictions
Click the "Predict" button to generate predictions.
The results will be displayed in a styled container:
Linear Regression: Displays the predicted number of attendees.
Logistic Regression: Displays the probability of attendance as a percentage.

Key Components
1. Model Loading
The load_model function dynamically loads the appropriate model based on the user's selection:

2. Dataset Loading
The app loads and preprocesses datasets for both models:

Linear Regression: university_event_turnout_data.csv
Logistic Regression: student_event_attendance.csv (with unnecessary columns removed).
3. Dynamic Input Generation
The app dynamically generates input fields based on the dataset's features:

3. Dynamic Input Generation
The app dynamically generates input fields based on the dataset's features:

4. Prediction and Display
The app uses the selected model to make predictions and displays the results:

Example Scenarios
Scenario 1: Predict Event Attendance
Select Linear Regression from the sidebar.
Enter the required numeric and categorical inputs.
Click Predict to view the predicted number of attendees.
Scenario 2: Predict Likelihood of Attendance
Select Logistic Regression from the sidebar.
Enter the required inputs (e.g., student demographics, event details).
Click Predict to view the likelihood of attendance as a percentage.
Troubleshooting
Common Issues
Model Loading Error:

Ensure the models directory contains lr_pipeline.joblib and log_pipeline.joblib.
Verify that the models were saved with a compatible version of scikit-learn.
Dataset Not Found:

Ensure the data/ directory contains the required datasets.
Verify the file paths in app.py.
Streamlit Not Installed:

Install Streamlit using:  pip install streamlit

Future Improvements
Add support for additional models.
Implement advanced error handling for invalid inputs.
Enhance the UI with more interactive visualizations.
Allow users to upload custom datasets for predictions.
License
This project is licensed under the MIT License.

Acknowledgments
Streamlit: For providing an easy-to-use framework for building web apps.
scikit-learn: For machine learning model training and deployment.
Pandas: For data manipulation and preprocessing.

