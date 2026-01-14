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


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process the property listings data.

    The steps taken are the following.
    1.  Remove listings that are not single-family homes.
    2.  Drop listings which are missing either their square footage or lot size.
    3.  Reset the indexing, so that there are no gaps in indexing
        after rows have been dropped.
    4.  Rename the columns.
    5.  Expand the `features` columns, one-hot encoding the results.
    6.  Remove the listings whose `features_unitCount` is greater than 1
        (as they are not single-family homes).
    7.  [Once again] Reset the indexing, so that there are no gaps in indexing
        after rows have been dropped.
    8.  For each numeric column that will be used as a prediction feature,
        replace the missing values with zeroes.
    9.  Replace missing `yearBuilt` entries with the median year of construction.
    10. Convert the sale data to a `pd.Period` format.

    Once all the steps are carried out, the modified `data` `DataFrame` is returned.
    """
    _focus_in_single_family_homes(data)
    _drop_listings_with_missing_sizes(data)
    _reset_index_after_dropping_rows(data)
    _rename_columns(data)
    data = _expand_features(data)
    _remove_listings_with_high_unit_count(data)
    _reset_index_after_dropping_rows(data)
    _fill_missing_numeric_values_with_zeroes(data)
    _fill_missing_year_built_with_median(data)
    _convert_sale_date_type(data)
    return data


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


def _drop_listings_with_missing_sizes(data: pd.DataFrame) -> None:
    """
    Remove listings whose `squareFootage` or `lotSize` entry is missing.

    This is done in-place.
    """
    data.dropna(subset=["squareFootage", "lotSize"], how="any", inplace=True)


def _reset_index_after_dropping_rows(data: pd.DataFrame) -> None:
    """
    We reset, in-place, the index of the `data` `DataFrame`.
    This is convenient to do after dropping rows from the `DataFrame`
    since it removes any gaps in the indexing of the `DataFrame`.
    """
    data.reset_index(inplace=True)
    del data["index"]


def _rename_columns(data: pd.DataFrame, columns=None) -> None:
    """
    Rename the columns of the `data` `DataFrame` in-place.
    """
    if columns is None:
        columns = {
            "lastSalePrice": "price",
            "squareFootage": "sqFt",
            "lastSaleDate": "saleDate",
        }
    data.rename(columns=columns, inplace=True)


def _expand_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Each entry of the `features` column of the `data` `DataFrame` is a dictionary.
    This is because the data comes from a `json` file, which allows non-homogeneous
    hierarchies of columns.

    The keys of these dictionaries are different types of features,
    such as `exteriorType` or `roofType`. The values are the feature that particular
    property has, e.g. `Brick` or `Slate`, respectively.

    Here we expand these features, such that the `features` column is removed and,
    in its stead, one-hot encoded columns of the form `features_exteriorType_Brick` and
    `features_roofType_Slate` are created.

    Once all of this is done, the updated `data` `DataFrame` is returned.
    """
    # `pd.get_dummies` automatically detects which columns contain numeric data and
    # it leaves these columns unmodified.
    features = pd.get_dummies(pd.json_normalize(data["features"]), dummy_na=True)
    # Remove non-alphanumeric characters from the resulting column names
    features.columns = features.columns.str.replace(r"\W", "", regex=True)
    features = features.add_prefix("features_")
    data = pd.concat([data, features], axis=1)
    del features, data["features"]
    return data


def _remove_listings_with_high_unit_count(data: pd.DataFrame) -> None:
    """
    The newly created `features_unitCount` column sometimes contains values
    greater than 1. This indicates that the home is not used as a single-family homes,
    and so such homes are dropped from the data.

    This is done in-place.
    """
    data.drop(data[data["features_unitCount"] > 1].index, inplace=True)


def _fill_missing_numeric_values_with_zeroes(
    data: pd.DataFrame, numeric_columns=None
) -> None:
    """
    Some numeric columns, which will be used as prediction features, are missing values.
    This function replaces the missing values with zeroes.

    For each such `column`, a sentinel column `column_nan` is created to indicate
    that, in the transformed `column`, the zero value comes from a missing value.

    This is done in-place.
    """
    if numeric_columns is None:
        numeric_columns = [
            "bedrooms",
            "bathrooms",
            "features_floorCount",
            "features_garageSpaces",
            "features_roomCount",
        ]
    data[[column + "_nan" for column in numeric_columns]] = data[numeric_columns].isna()
    data.fillna(value={column: 0 for column in numeric_columns}, inplace=True)


def _fill_missing_year_built_with_median(data: pd.DataFrame) -> None:
    """
    Replaces the missing values in the `yearBuilt` column with the *median*
    among the non-missing values in that same column.

    Create a sentinel column `yearBuilt_nan` which indicates when the `yearBuild` value
    was initially missing.

    This is done in-place.
    """
    data["yearBuilt_nan"] = data["yearBuilt"].isna()
    data.fillna(value={"yearBuilt": data["yearBuilt"].median()}, inplace=True)


def _convert_sale_date_type(data: pd.DataFrame) -> None:
    """
    Convert the sale date to the `datetime` format, remove the timezone information, and
    keep only the month and year. This means that the final format is `pd.Period`.

    This is done in-place.
    """
    data["saleDate"] = pd.to_datetime(data["saleDate"])
    data["saleDate"] = data["saleDate"].dt.tz_localize(None)
    data["saleDate"] = data["saleDate"].dt.to_period("M")
