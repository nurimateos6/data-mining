import urllib.request
from io import StringIO
import pandas as pd


def get_data(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data = response.read()
    return data


def bytes_to_df(txt: bytes):
    txt = str(txt, 'utf-8')
    data = StringIO(txt)
    cols_df = ["GridID", "date", "Shift", "Accident", "Longitude.grid", "Latitude.grid"]
    df = pd.read_csv(data, usecols=cols_df, sep=',')
    return df


def load_df_data(url):
    data = get_data(url)
    df = bytes_to_df(data)
    return df
