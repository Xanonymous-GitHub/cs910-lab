from numpy import where, log
from pandas import DataFrame
from patsy.highlevel import dmatrices
from sklearn.metrics import accuracy_score
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS

# Dummy assignment to avoid `imported but unused` error,
# Since the `log` function is used in the `dmatrices` function.
_LOG = log


def create_regression_stat_model(
        data: DataFrame,
        equation: str,
):
    data.dropna(inplace=True)
    y, x = dmatrices(equation, data=data, return_type='dataframe')
    model = OLS(y, x)
    return model.fit()


def create_logistic_stat_model(
        data: DataFrame,
        equation: str,
):
    data.dropna(inplace=True)
    y, x = dmatrices(equation, data=data, return_type='dataframe')
    model = Logit(y, x)
    result = model.fit(disp=False)
    predicted = model.predict(result.params)

    # noinspection PyTypeChecker
    predict_class = where(predicted > 0.5, 1, 0)

    accuracy = accuracy_score(y, predict_class)
    return result, accuracy
