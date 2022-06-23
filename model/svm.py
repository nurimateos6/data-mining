from validation.validation import scoring
from sklearn.svm import SVC
import pandas as pd


def svm(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    SVM = SVC(kernel='linear', class_weight='balanced')
    SVM.fit(X_train,y_train)
    y_pred = SVM.predict(X_test)
    score, confusion_matrix, especifidad, sensibilidad = scoring(X_test, y_test, SVM, y_pred)
    return score, confusion_matrix, especifidad, sensibilidad