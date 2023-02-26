import os
from data.load_data import load_df_data

data = os.environ['INPUT_DATA']


def main():
    df = load_df_data(data)

