"""
This module contains methods used to evaluate the various models.
"""

import numpy as np
import pandas as pd


def hpi_mse(property_listings: pd.DataFrame, target: str = "price") -> float:
    """
    This method expects a `DataFrame` `property_listings` with the following columns:
    - `price`, the sale price, or
    - `logPrice`, the logarithm of the sale price, and
    - `trueValueHomePriceIndex`, the value of the home price index on the month of
      the sale, and
    - `availableValueHomePriceIndex`, the value of the home price index *available*
      on the month of the sale (typically this is the home price index 3-month prior
      since the home price index used here is published with a 3-month lag).

    It computes the mean squared error inherent to using the *available* home price
    index instead of the *true* home price index. This is, roughly speaking, a measure
    of the error coming from using an index with a 3-month lag.

    The `target` argument is either `price` or `logPrice`, depending on whether
    the error in prices or log-prices is to be computed.
    """
    truth = property_listings["price"]
    estimate = property_listings["price"] * (
        property_listings["availableValueHomePriceIndex"]
        / property_listings["trueValueHomePriceIndex"]
    )
    if target == "logPrice":
        truth, estimate = np.log(truth), np.log(estimate)
    error = truth - estimate
    return np.mean(error**2)


def hpi_rmse(property_listings: pd.DataFrame, target: str = "price") -> float:
    """
    This method expects a `DataFrame` `property_listings` with the following columns:
    - `price`, the sale price, or
    - `logPrice`, the logarithm of the sale price, and
    - `trueValueHomePriceIndex`, the value of the home price index on the month of
      the sale, and
    - `availableValueHomePriceIndex`, the value of the home price index *available*
      on the month of the sale (typically this is the home price index 3-month prior
      since the home price index used here is published with a 3-month lag).

    It computes the root mean squared error inherent to using the *available* home
    price index instead of the *true* home price index. This is, roughly speaking,
    a measure of the error coming from using an index with a 3-month lag.

    The `target` argument is either `price` or `logPrice`, depending on whether
    the error in prices or log-prices is to be computed.
    """
    return np.sqrt(hpi_mse(property_listings, target))
