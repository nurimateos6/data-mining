from sklearn.model_selection import train_test_split
from etl.cleaners import to_numerical
import pandas as pd


def split_label(df: pd.DataFrame):
    X = df.loc[:, ~df.columns.isin(['Accident'])]
    y = df['Accident']
    return X, y


def split_data(df: pd.DataFrame):
    df = to_numerical(df)
    X, y = split_label(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test
