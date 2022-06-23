import pandas as pd
from sklearn.preprocessing import LabelEncoder


def col_to_datetime(df: pd.DataFrame, col):
    # return pd.to_datetime(df[col])
    df[col] = df[col].apply(pd.to_datetime)
    df['year'] = df[col].map(lambda x: x.year)
    df['month'] = df[col].map(lambda x: x.month)
    df['day'] = df[col].map(lambda x: x.day)
    del df[col]


def encoder_labels(df: pd.DataFrame, col):
    le = LabelEncoder()
    le.fit(df[col])
    values = le.transform(df[col])
    values = pd.Series(values)
    df[col] = values


def to_numerical(df: pd.DataFrame):
    l_cols = [col for col in df.columns if df[col].dtype == 'O']
    if len(l_cols) > 0:
        col_to_datetime(df, l_cols[0])
        encoder_labels(df, l_cols[1])
    return df

