"""
This module contains the methods used to load and pre-process the property listings data
obtained via the `properties` endpoints of the RentCast API
(see https://developers.rentcast.io/reference/property-data).
"""

from importlib import resources

import numpy as np
import pandas as pd


def load_demo_data(path: str = None) -> pd.DataFrame:
    """
    Load the raw property listings data from the `json` file obtained from the RentCast
    API into a `pandas` `DataFrame`.
    """
    if path is None:
        path = resources.files("hopus").joinpath("demo_data/data_v1.json")
    return pd.read_json(path)


def preprocess(
    property_listings_data: pd.DataFrame, home_price_index_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Pre-process the property listings data.

    Note that the home price index data is also required, since it is eventually merged
    with the property listings data.

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
    10. Convert the sale date to a `pd.Period` format.
    11. Split the sale date into month and year (as separate columns).
    12. Merged with the (already processed) home price index data.
    13. Compute the 'time-normalized price-per-square-foot' (obtained by dividing
        the price-per-square-foot by the current value of the home price index).
    14. Compute the logarithms of the sale prices.

    Once all the steps are carried out, the modified `property_listings`
    `DataFrame` is returned.
    """
    # Step 1
    _focus_in_single_family_homes(property_listings_data)
    # Step 2
    _drop_listings_with_missing_sizes(property_listings_data)
    # Step 3
    _reset_index_after_dropping_rows(property_listings_data)
    # Step 4
    _rename_columns(property_listings_data)
    # Step 5
    property_listings_data = _expand_features(property_listings_data)
    # Step 6
    _remove_listings_with_high_unit_count(property_listings_data)
    # Step 7
    _reset_index_after_dropping_rows(property_listings_data)
    # Step 8
    _fill_missing_numeric_values_with_zeroes(property_listings_data)
    # Step 9
    _fill_missing_year_built_with_median(property_listings_data)
    # Step 10
    _convert_sale_date_type(property_listings_data)
    # Step 11
    _split_sale_date(property_listings_data)
    # Step 12
    property_listings_data = _merge_with_home_price_index_data(
        property_listings_data, home_price_index_data
    )
    # Step 13
    _compute_time_normalized_price_per_square_foot(property_listings_data)
    # Step 14
    _compute_log_price(property_listings_data)
    return property_listings_data


def drop_outliers(data: pd.DataFrame, cutoff: tuple[float, float] = (0.2, 2.0)) -> None:
    """
    We use the `timeNormalizedPricePerSqFt' column to detect, and drop, outliers.

    These are property listings where the sale price is, to be informal, *surprising*.
    It can be surprisingly low or surprisingly high, and typically this is due to
    factors that are not directly accessible in the data. For example a property may
    be sold at a very low price because the previous owner and occupant passed away, or
    it may be sold at a very high price because the house comes with a gorgeous view
    overlooking a lake.

    The default cutoffs chosen at 0.2 and 2.0 are arbitrary.
    On the preliminary data they remove about a dozen outliers each,
    out of about 1,700 samples.

    This is done in-place and the index is reset after the rows are dropped.
    """
    low_cutoff, high_cutoff = cutoff
    data.drop(
        data[
            (data["timeNormalizedPricePerSqFt"] < low_cutoff)
            | (data["timeNormalizedPricePerSqFt"] > high_cutoff)
        ].index,
        inplace=True,
    )
    _reset_index_after_dropping_rows(data)


def drop_missing_key_features(
    data: pd.DataFrame, column_to_group_map_path: str = None
) -> None:
    """
    We use an external `csv` file to label certain columns as *key* prediction features.
    All rows which are missing data in at least one of these columns is then dropped.

    This is done inplace and the index is reset after the rows are dropped.
    """
    if column_to_group_map_path is None:
        column_to_group_map_path = resources.files("hopus").joinpath(
            "config/column_to_group_map.csv"
        )

    column_to_group_map = pd.read_csv(column_to_group_map_path)
    column_to_group_map = dict(
        zip(column_to_group_map["Key"], column_to_group_map["Value"])
    )
    for column in data.columns:
        if (
            column in column_to_group_map
            and column_to_group_map[column] == "keyPredictionFeatures"
            and (column + "_nan") in data.columns
        ):
            data.drop(data[data[column + "_nan"] == 1].index, inplace=True)
            del data[column + "_nan"]
    _reset_index_after_dropping_rows(data)


def group_columns(
    data: pd.DataFrame,
    column_to_group_map_path: str = "config/column_to_group_map.csv",
) -> None:
    """
    Group the property listings data columns into four categories:
    - `identification` (e.g. the address, ZIP code, etc.),
    - `keyPredictionFeatures` (e.g. the square footage, number of bedrooms, etc.),
    - `auxiliaryPredictionFeatures` (e.g. whether there is a garage, a fireplace, etc.),
    - `target` (i.e. the price), and
    - `unused` (for miscellaneous columns, such as the `zoning` column).

    The map which takes a column name to a category is encoded externally, in the `csv`
    file `column_to_group_map_path`, except for the `features_` columns:
    they are all mapped to the `predictionFeatures` group
    (except for `features_unitCount` which is mapped to `unused`).

    This is done in-place.
    """

    # Build the multi_index_map
    column_to_group_map = pd.read_csv(column_to_group_map_path)
    multi_index_map = dict(
        zip(column_to_group_map["Key"], column_to_group_map["Value"])
    )
    data.columns = pd.MultiIndex.from_arrays(
        [[multi_index_map[column] for column in data.columns], data.columns]
    )


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


def _split_sale_date(data: pd.DataFrame) -> None:
    """
    Split the sale date into two pieces: the month and the year.

    This is done in-place.
    """
    data["saleMonth"] = data["saleDate"].dt.month
    data["saleYear"] = data["saleDate"].dt.year


def _merge_with_home_price_index_data(
    property_listings_data: pd.DataFrame, home_price_index_data
) -> pd.DataFrame:
    """
    Merge the property listings data and the home price index data using the date
    of sale as key on which to join these two DataFrames.

    Returned the merged `DataFrame`.
    """
    return pd.merge(
        left=property_listings_data,
        right=home_price_index_data.add_suffix("HomePriceIndex"),
        left_on="saleDate",
        right_index=True,
        how="inner",
    )


def _compute_time_normalized_price_per_square_foot(data: pd.DataFrame) -> None:
    """
    Compute the price-per-square-foot and the time-normalized price-per-square-foot,
    which is the price-per-square-foot divided by the home price index.

    This can be used to detect outliers.

    This is done in-place.
    """
    data["pricePerSqFt"] = data["price"] / data["sqFt"]
    data["timeNormalizedPricePerSqFt"] = (
        data["pricePerSqFt"] / data["trueValueHomePriceIndex"]
    )


def _compute_log_price(data: pd.DataFrame) -> None:
    """
    Use the `price` column to compute the `logPrice` column.

    This is done in-place.
    """
    data["logPrice"] = np.log(data["price"])
