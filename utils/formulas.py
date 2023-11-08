from numbers import Real
from typing import Sequence

from numpy import polyfit


def calculate_linear_regression_parameters[T: Real](
        x_series: Sequence[T],
        y_series: Sequence[T],
        /,
) -> tuple[T, T]:
    """
    Calculate the linear regression parameters (slope and intercept) for the given data series.
    Args:
        x_series: x series
        y_series: y series
    Returns:
        two coefficients of the regression, slope and intercept.
    """

    # Destruct the np.ndarray to isolated variables.
    m, b = polyfit(x_series, y_series, 1)
    return m, b
