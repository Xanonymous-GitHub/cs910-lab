from pandas import DataFrame
from patsy.highlevel import dmatrices
from statsmodels.regression.linear_model import OLS
from numpy import log


def create_regression_stat_model(
        data: DataFrame,
        equation: str,
):
    data.dropna()
    y, x = dmatrices(equation, data=data, return_type='dataframe')
    model = OLS(y, x)
    return model.fit()
