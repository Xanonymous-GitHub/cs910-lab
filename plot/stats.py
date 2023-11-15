from pandas import DataFrame
from patsy.highlevel import dmatrices
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS


def create_regression_stat_model(
        data: DataFrame,
        equation: str,
):
    data.dropna()
    y, x = dmatrices(equation, data=data, return_type='dataframe')
    model = OLS(y, x)
    return model.fit()


def create_logistic_stat_model(
        data: DataFrame,
        equation: str,
):
    data.dropna()
    y, x = dmatrices(equation, data=data, return_type='dataframe')
    model = Logit(y, x)
    return model.fit()
