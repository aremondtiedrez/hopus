"""
This module contains methods which pre-process real estate data.

More specifically, this module expects data to come from two sources.
- Home price index data, a `csv` file containing the S&P Cotality Case-Shiller
  U.S. National Home Price Index. This can be downloaded, for example, from
  the Federal Reserve Bank of St. Louis: https://fred.stlouisfed.org/series/CSUSHPINSA
- Real estate listings, a `json` file containing public record data for residential
  properties. This is obtained via the `properties` endpoint of the RentCast API
  (see https://developers.rentcast.io/reference/property-data).
"""

import pandas as pd


class HomePriceIndex:
    """
    This class contains the methods used to load and pre-process the home price index
    data.
    """

    def load(self, path: str = "data/CSUSHPINSA.csv"):
        """
        Load the raw home price index data from a `csv` file
        into a `pandas` `DataFrame`.
        """
        return pd.read_csv(path)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-process the home price index data.

        The steps taken are the following.
        1. Rename the columns of the `DataFrame`.
        """
        self._rename_columns(data)

    def _rename_columns(self, data: pd.DataFrame, columns=None):
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
