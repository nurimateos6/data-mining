from sklearn.neural_network import MLPClassifier
from validation.validation import scoring
import pandas as pd


def mlp(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score, confusion_matrix, especifidad, sensibilidad = scoring(X_test, y_test, clf, y_pred)
    return score, confusion_matrix, especifidad, sensibilidad
