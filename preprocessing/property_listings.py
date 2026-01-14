"""
This module contains the methods used to load and pre-process the property listings data
obtained via the `properties` endpoints of the RentCast API
(see https://developers.rentcast.io/reference/property-data).
"""

import pandas as pd


def load(path: str = "data/data_v1.json") -> pd.DataFrame:
    """
    Load the raw property listings data from the `json` file obtained from the RentCast
    API into a `pandas` `DataFrame`.
    """
    return pd.read_json(path)


def preprocess(data: pd.DataFrame) -> None:
    """
    Pre-process the property listings data.

    The steps taken are the following.
    1. Remove listings that are not single-family homes.
    """
    _focus_in_single_family_homes(data)


def _focus_in_single_family_homes(data: pd.DataFrame) -> None:
    """
    Remove listings whose `propertyType` is not `Single Family`,
    i.e. remove listings that are not single-family homes.

    The column `propertyType` is also removed,
    since it now trivial and no longer needed.

    This is done in-place.
    """
    data.drop(data[data["propertyType"] != "Single Family"].index, inplace=True)
    del data["propertyType"]
