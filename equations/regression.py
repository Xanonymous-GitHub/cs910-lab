from collections.abc import Sequence
from numbers import Real

from numpy import polyfit, ndarray, float64


def calculate_linear_regression_parameters(
        *,
        x_series: Sequence[Real] | ndarray[Real],
        y_series: Sequence[Real] | ndarray[Real],
) -> tuple[float64, float64]:
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
