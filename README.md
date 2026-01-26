# HOPUS

**HOusing Pricing UtilitieS (HOPUS)**: this repository contains utilities for the prediction
of real estate listings sales prices.

## \>\>\> Start here! \<\<\<
The easiest way to see what HOPUS can do is to peruse this [notebook](
https://colab.research.google.com/github/aremondtiedrez/hopus/blob/main/demo.ipynb
) (via Google Colab).
A *non-interactive* version of this notebook is also accessible more readily via
[GitHub](https://github.com/aremondtiedrez/hopus/blob/main/demo.ipynb).
This notebook leverages HOPUS utilities to
- clean the raw data,
- train a variety of models for the prediction of real estate prices,
- evaluate the performance of these models, and
- display the model predictions on a geographical map.

The model obtained is used to display geographically
its predictions side-by-side with the true prices
on this [interactive map](https://aremondtiedrez.github.io/hopus/predictions_map.html).

## Where does the data come from?

The data used to train the models therein was obtained
via the [RentCast API](https://www.rentcast.io/api).
