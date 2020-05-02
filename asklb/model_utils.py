"""
Contains model fitting and performance functions
"""

from autosklearn.classification import AutoSklearnClassifier
import pandas as pd


def process_feat_types(data):
    """
    Factorizes categorical columns in data and returns feat_types for
    auto-sklearn input.

    Args:
        data (pd.DataFrame): the input data

    Returns:
        tuple:
        pd.DataFrame: data with factorized categorical features
        list: list of feature types
        dict: dict with (col, [categorical variables]) pairs
    """

    # assumes all categorical features are objects
    feat_types = ["Categorical" if x == object else "Numerical" for x in data.dtypes]

    cat_dict = {}
    for i, col in enumerate(data.columns):
        if feat_types[i] == 'Categorical':
            factored_col, idx = data[col].factorize()
            data[col] = factored_col
            cat_dict[col] = list(idx)

    return data, feat_types, cat_dict