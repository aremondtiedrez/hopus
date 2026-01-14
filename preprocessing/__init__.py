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

from . import home_price_index
