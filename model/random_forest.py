from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from validation.validation import scoring
import pandas as pd


def rf(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    clf = RandomForestClassifier(max_depth=50, random_state=42, class_weight='balanced_subsample', n_estimators=50, criterion='gini')
    clf.fit(X_train, y_train)
    adaboost = AdaBoostClassifier(base_estimator = clf, random_state=42, learning_rate=0.1)
    adaboost.fit(X_train, y_train)
    y_pred = adaboost.predict(X_test)
    score, confusion_matrix, especifidad, sensibilidad = scoring(X_test, y_test, adaboost, y_pred)
    return score, confusion_matrix, especifidad, sensibilidad