from model_s.linear_regression import LinearRegressionCustom
from model_s.logistics_regression import LogisticRegressionCustom
from preprocessing.data_cleaning import load_and_preprocess
from evaluation.metrics import evaluate_classification, evaluate_regression

# Load and preprocess
X_train, X_test, y_train, y_test = load_and_preprocess('event_data.csv')

# Train Linear Regression
lin_reg = LinearRegressionCustom()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
print("Linear Regression MSE:", evaluate_regression(y_test, y_pred_lin))

# Train Logistic Regression
log_reg = LogisticRegressionCustom()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
cm, acc = evaluate_classification(y_test, y_pred_log)
print("Logistic Regression Accuracy:", acc)
print("Confusion Matrix:\n", cm)
