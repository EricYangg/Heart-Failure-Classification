import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate

# Title: Function to consolidate cross validation scores into a pandas series.
# Author: Varada Kolhatkar & Michael Gelbart
# Source: https://pages.github.ubc.ca/mds-2025-26/DSCI_571_sup-learn-1_students/README.html 
# Taken from: DSCI-571: Laboratory 2

def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    # --- Input Validation Check ---
    # Check: Model has fit method
    if not hasattr(model, "fit"):
        raise TypeError("Input model must implement a 'fit' method")

    # Check: X_train input
    if not isinstance(X_train, (pd.DataFrame, np.ndarray)):
        raise TypeError("Input X_train must be a numpy array or pandas DataFrame")

    if X_train is None or len(X_train) == 0:
        raise ValueError("Input X_train must contain observations")

    X_arr = np.asarray(X_train)
    if X_arr.ndim != 2 or X_arr.shape[1] < 1:
        raise ValueError("Input X_train must have at least one feature column")

    # Check: y_train input
    if not isinstance(y_train, (pd.Series, np.ndarray, list)):
        raise TypeError("Input y_train must be a pandas Series, numpy array, or list")

    y_arr = np.asarray(y_train)
    if y_arr.ndim != 1:
        raise ValueError("Input y_train must be 1-dimensional")
    if y_arr.size == 0:
        raise ValueError("Input y_train must contain observations")
    
    # Check: Matching data sets lenght
    if len(X_train) != len(y_train):
        raise ValueError("Input X_train and y_train must have the same number of rows")
    
    # --- Cross Validation Execution ----
    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores.iloc[i], 
                                                std_scores.iloc[i])))

    return pd.Series(data=out_col, index=mean_scores.index)