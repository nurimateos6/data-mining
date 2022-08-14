from sklearn.linear_model import LogisticRegression
from validation.validation import scoring
import pandas as pd

def logistic_regression(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    lr = LogisticRegression(random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    score, confusion_matrix, especifidad, sensibilidad = scoring(X_test, y_test, lr, y_pred)

    return score, confusion_matrix, especifidad, sensibilidad


