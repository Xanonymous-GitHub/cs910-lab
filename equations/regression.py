from collections.abc import Sequence
from numbers import Real

from numpy import polyfit, ndarray, float64
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


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


def calculate_multiple_linear_regression_parameters(
        *,
        x_series: Sequence[Sequence[Real]] | ndarray[ndarray[Real]],
        y_series: Sequence[Real] | ndarray[Real],
        test_ratio: float = 0.3,
) -> tuple[ndarray[float64], float64, float]:
    """
    Calculate the linear regression parameters (slope and intercept) for the given data series.
    Args:
        x_series: x series
        y_series: y series
        test_ratio: test ratio
    Returns:
        The test data, the predicted data, and the score.
    """
    x_trains, x_tests, y_trains, y_tests = train_test_split(
        list(zip(*x_series)),
        y_series,
        test_size=test_ratio,
        random_state=0
    )

    lr = LinearRegression()
    lr.fit(x_trains, y_trains)

    y_predict = lr.predict(x_tests)
    score = r2_score(y_tests, y_predict)

    return y_tests, y_predict, score
