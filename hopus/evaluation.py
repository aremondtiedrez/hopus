"""
This module contains methods used to evaluate the various models.
"""

import secrets

from sklearn.model_selection import KFold

import numpy as np
import pandas as pd

from . import models


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


def cv_evaluation(  # pylint: disable=too-many-locals, too-many-arguments
    model_class: models.Model,
    features: pd.DataFrame,
    target: pd.Series,
    n_splits: int = 5,
    seed: int = 2026,
    hyperparameters: dict = None,
    **kwargs,
) -> tuple[float, float, list[models.Model]]:
    """
    Performs cross-validation on a `model` using the `features` and `target` provided.
    This method expects `features` and `target` to share the same index.

    The split into folds is governed by `n_splits`, the number of folds to split
    the data into, and `seed`, an integer which is used as the random seed for
    the random splitting.

    Returns
    train_cv_mse    Average mean squared error over the training folds.
    test_cv_mse     Average mean squared error over the testing folds.
    trained_models  List of trained models, each of which is
                    an instance of `models.Model`.
    """
    # Input validation
    if hyperparameters is None:
        hyperparameters = {}

    # Split the data (virtually, i.e. by splitting the indices)
    # This is where it is essential that the `features` and `target`
    # share the same index.
    fold_indices = list(
        KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(features)
    )

    # Train and evaluate the models
    trained_models = []
    squared_errors = np.zeros(shape=(n_splits, 2))
    for fold_number, (train_indices, test_indices) in enumerate(fold_indices):

        # Training
        model = model_class(**hyperparameters)
        trained_models.append(model)
        model.fit(features.iloc[train_indices], target.iloc[train_indices])

        # Evaluation
        train_mse = model.evaluate(
            features.iloc[train_indices], target.iloc[train_indices], **kwargs
        )
        test_mse = model.evaluate(
            features.iloc[test_indices], target.iloc[test_indices], **kwargs
        )
        squared_errors[fold_number] = train_mse, test_mse

    train_cv_mse, test_cv_mse = squared_errors.mean(axis=0)

    return train_cv_mse, test_cv_mse, trained_models


def run_experiment(  # pylint: disable=too-many-arguments
    features: pd.DataFrame,
    target: pd.DataFrame,
    model_class: models.Model,
    hyperparameters: dict,
    n_experiments: int,
    n_splits: int,
) -> dict:
    """
    Given a model class, hyperparameters, and experiment parameters, train models
    of that class and with these parameters. The experiment parameters are `n_splits`
    and `n_experiments`, which set the number of cross-validation folds and the number
    of experiments to run, respectively.

    Arguments
    features                The inputs of the model.
    target                  The true target outputs.
    model_class             The class of model to train (from `models`).
    hyperparameters         A dictionary of hyperparameters. The keys of this dictionary
                            must be the names of keyword arguments that can be passed to
                            the given model class.
    n_experiments           The number of experiments to run (each will use a random
                            seed for splitting the data into cross-validation folds).
    n_splits                The number of cross-validation folds used.

    Returns
    record                  A dictionary whose keys are the hyperparameters,
                            experiment parameters, and experiment result names and
                            the values are their corresponding values.
    """
    experiment_parameters = {"n_splits": n_splits, "seed": None}
    for _ in range(n_experiments):
        experiment_parameters["seed"] = secrets.randbits(32)
        train_cv_mse, test_cv_mse, _ = cv_evaluation(
            model_class,
            features,
            target,
            **experiment_parameters,
            hyperparameters=hyperparameters,
        )
        experiment_result = {
            "train_cv_mse": train_cv_mse,
            "test_cv_mse": test_cv_mse,
        }
        record = {
            **experiment_parameters,
            **hyperparameters,
            **experiment_result,
        }
    return record
