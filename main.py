from data.load_data import load_df_data
from featuring.split_data import split_data
from model.logistic_regression import logistic_regression
from model.svm import svm
from model.random_forest import rf
from model.nn import mlp


if __name__ == "__main__":
    df = load_df_data("https://raw.githubusercontent.com/4data-lab/datasets/master/exercise.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    score, confusion_matrix, especifidad, sensibilidad = logistic_regression(X_train, y_train, X_test, y_test)
    score1, confusion_matrix1, especifidad1, sensibilidad1 = svm(X_train, y_train, X_test, y_test)
    score2, confusion_matrix2, especifidad2, sensibilidad2 = rf(X_train, y_train, X_test, y_test)
    score3, confusion_matrix3, especifidad3, sensibilidad3 = mlp(X_train, y_train, X_test, y_test)