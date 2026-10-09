from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
import numpy as np
def evaluate_classification(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return cm, acc

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse



def mean_squared_error(self, true_y, predicted_y):
        mse = np.mean((true_y - predicted_y) ** 2)
        return mse
