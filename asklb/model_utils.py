"""
Contains model fitting and performance functions.
"""

from autosklearn.classification import AutoSklearnClassifier
import numpy as np
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


def thresholdout(train_acc, test_acc, threshold=0.01, noise=0.03):
    """
    Applies the thresholdout algorithm to produce a (possibly) noised output.
    
    An implementation of the algorithm presented in Section 3 of "Holdout Reuse."
    
    Args:
        train_acc (float)
        test_acc (float)
        threshold (float): the base difference between train and test accuracies for thresholdout to apply
        noise (float): the noise rate for the Laplacian noise applied

    Returns:
        float: potentially noised test accuracy
    """
    threshold_hat = threshold + np.random.laplace(0, 2*noise)
    
    if np.abs(train_acc - test_acc) > (threshold_hat + np.random.laplace(0, 4*noise)):
        return np.clip(test_acc + np.random.laplace(0, noise), 0, 1)
    else:
        return train_acc