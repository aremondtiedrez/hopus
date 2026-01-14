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
    2. Convert the `date` column to `datetime`, keep only the month and year, and
       use the date as the new index.
    """
    _rename_columns(data)
    _convert_date_type(data)


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


def _convert_date_type(data: pd.DataFrame):
    """
    Convert the type of the `data` column to be a `datetime` object,
    then keep only the month and the year of that data,
    and finally set that column to be the index of the `DataFrame`.

    All of this is done in-place.
    """
    data["date"] = pd.to_datetime(data["date"])
    data["date"] = data["date"].dt.to_period("M")
    data.set_index("date", inplace=True)
