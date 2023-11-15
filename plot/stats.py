from pandas import DataFrame
from patsy.highlevel import dmatrices
from statsmodels.regression.linear_model import OLS


def create_regression_stat_model(
        data: DataFrame,
        y_column: str,
        *x_columns: str,
):
    data.dropna()
    x_formula = ' + '.join(x_columns)
    y, x = dmatrices(f'{y_column} ~ {x_formula}', data=data, return_type='dataframe')
    model = OLS(y, x)
    return model.fit()
