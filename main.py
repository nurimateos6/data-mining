import os
from data.load_data import load_df_data
from featuring.split_data import split_data
from model.logistic_regression import logistic_regression
from model.svm import svm
from model.random_forest import rf
from model.nn import mlp
import sys

data = os.environ['INPUT_DATA']

def main():
    arg = sys.argv[1]
    if arg == '--load_data':
        df = load_df_data(data)

    if arg == '--run_all':
        df = load_df_data(data)
        X_train, X_test, y_train, y_test = split_data(df)
        score2, confusion_matrix2, especifidad2, sensibilidad2 = rf(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
