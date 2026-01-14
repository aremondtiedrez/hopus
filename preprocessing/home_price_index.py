"""
This module contains the methods used to load and pre-process the home price index data.
"""

import pandas as pd


def load(path: str = "data/CSUSHPINSA.csv"):
    """
    Load the raw home price index data from a `csv` file
    into a `pandas` `DataFrame`.
    """
    return pd.read_csv(path)


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process the home price index data.

    The steps taken are the following.
    1. Rename the columns of the `DataFrame`.
    """
    _rename_columns(data)


def _rename_columns(data: pd.DataFrame, columns=None):
    """
    Rename the columns of the home price index `DataFrame` in-place.
    The default argument maps the default column names found in the `csv` file
    from which the home price index data is pulled to column names that are chosen
    to make the remaining pre-processing steps easier to follow.
    """
    if columns is None:
        columns = {
            "observation_date": "date",
            "CSUSHPINSA": "trueValue",
        }
    data.rename(columns=columns, inplace=True)
