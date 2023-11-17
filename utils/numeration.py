from pandas import DataFrame


def quantified_from(source: DataFrame, /, *columns: str) -> DataFrame:
    """
    Creates a new DataFrame with quantified values from the source DataFrame.
    The quantification is done by assigning a unique integer to each unique value in the source DataFrame.
    The resulting DataFrame will have the same index as the source DataFrame.
    The resulting DataFrame will have the same columns as the source DataFrame, but with the quantified columns
    replacing the original columns.

    :param source: The DataFrame to be quantified.
    :param columns: The columns to be quantified.
    :return: A new DataFrame with quantified values.
    """
    quantified = source.copy()
    quantified.dropna(inplace=True, ignore_index=True)
    for column in columns:
        quantified[column] = quantified[column].astype('category', copy=True).cat.codes
    return quantified


def binary_quantified_from(source: DataFrame, /, *, column: str, positive_when_equal_to: str) -> DataFrame:
    """
    Creates a new DataFrame with quantified values from the source DataFrame.
    The quantification is done by assigning 1 to each row where the value in the specified column is equal to the
    specified value, and 0 otherwise.
    The resulting DataFrame will have the same index as the source DataFrame.
    The resulting DataFrame will have the same columns as the source DataFrame, but with the quantified column
    replacing the original column.

    :param source: The DataFrame to be quantified.
    :param column: The column to be quantified.
    :param positive_when_equal_to: The value to be considered positive.
    :return: A new DataFrame with quantified values.
    """
    quantified = source.copy()
    quantified.dropna(inplace=True, ignore_index=True)
    quantified[column] = (
        quantified[column]
        .apply(lambda value: 1 if value == positive_when_equal_to else 0)
    )
    return quantified
