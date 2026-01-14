"""
This module contains the methods used to load and pre-process the home price index data.
"""

import pandas as pd


def load(path: str = "data/CSUSHPINSA.csv") -> pd.DataFrame:
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


def _rename_columns(data: pd.DataFrame, columns=None) -> None:
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


def _convert_date_type(data: pd.DataFrame) -> None:
    """
    Convert the type of the `data` column to be a `datetime` object,
    then keep only the month and the year of that data,
    and finally set that column to be the index of the `DataFrame`.

    All of this is done in-place.
    """
    data["date"] = pd.to_datetime(data["date"])
    data["date"] = data["date"].dt.to_period("M")
    data.set_index("date", inplace=True)


def _add_three_month_lagged_value(data: pd.DataFrame):
    """
    The home price index used, namely the S&P Cotality Case-Shiller U.S. National
    Home Price Index, is only available about three months after the fact. In other
    words the home price index for January is only published in April.
    Here we account for this by creating a three-month shifted copy of the index.

    The `data` `DataFrame` is expected to be indexed by a `datetime` object whose period
    is monthly (i.e. that object contains information about the month and year only) and
    to contain a single column called `trueValue`. The `trueValue` is the actual value
    of the home price index that month.

    We create a second column, called `availableValue`, which shows, for month M,
    the home price index publicly available at that time. In other words:

        availableValue(month M) = trueValue(month M-3).

    Moreover, we then remove all rows for which it is not possible to compute
    the `availableValue`, namely the first three rows, in chronological order.
    Since the home price index goes back to January 1987, this does not cause
    any issues as long as we only consider real estate transactions that occur
    during, or after, April 1987.

    These operations is carried out in-place.
    """
    data["availableValue"] = data["trueValue"].shirt(3)
    data.dropna(how="any", inplace=True)
