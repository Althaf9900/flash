import math
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def _handle_dataframe_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform error handling for DataFrame input and preprocess parameters.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame to check.

    Raises:
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If df is empty or None.
    """
    if df is None:
        raise ValueError("DataFrame object cannot be None")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Please provide a valid pandas DataFrame object")

    if df.empty:
        raise ValueError("DataFrame object cannot be empty")

def get_cat_col(df: pd.DataFrame, unique_value_threshold: Optional[int] = 12, ignore_cols: Optional[str] = None) -> List[str]:
    
    _handle_dataframe_errors(df)

    if ignore_cols and ignore_cols in df.columns:
        df = df.drop(ignore_cols, axis=1)

    categorical_features = []

    for column in df.columns:

        is_categorical = pd.api.types.is_categorical_dtype(df[column])
        is_bool = pd.api.types.is_bool_dtype(df[column])
        is_object = pd.api.types.is_object_dtype(df[column])

        if is_categorical or is_bool or is_object:
            categorical_features.append(column)

        elif pd.api.types.is_numeric_dtype(df[column]):
            if df[column].nunique() <= unique_value_threshold:
                categorical_features.append(column)

    return categorical_features

def get_num_col(df: pd.DataFrame, unique_value_threshold: Optional[int] = 12, ignore_cols: Optional[str] = None) -> List[str]:
    
    _handle_dataframe_errors(df)

    if ignore_cols and ignore_cols in df.columns:
        df = df.drop(ignore_cols, axis=1)

    numerical_features = []

    for column in df.select_dtypes(include=[np.number]):
        if df[column].nunique() > unique_value_threshold:
            numerical_features.append(column)

    return numerical_features

def get_binary_col(df: pd.DataFrame, categorical_feature_list: Optional[list] = None, ignore_cols: Optional[str] = None) -> List[str]:
    
    _handle_dataframe_errors(df)
    binary_features = []

    # If no categorical_feature_list is specified, check all columns in df
    if categorical_feature_list is None:
        categorical_feature_list = df.columns.tolist()

    # If ignore_cols is specified, add it to categorical_feature_list
    if ignore_cols:
        categorical_feature_list = [feature for feature in categorical_feature_list if feature != ignore_cols]

    categorical_feature_list = [feature for feature in categorical_feature_list if feature in df.columns]

    # Iterate through specified categorical features or all columns
    for feature in categorical_feature_list:
        if df[feature].nunique() == 2:  # Check for exactly two unique values
            binary_features.append(feature)

    return binary_features

def find_single_value_col(
    df: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
    remove: bool = False,
    ignore_cols: Optional[str] = None
) -> Union[List[str], pd.DataFrame]:

    _handle_dataframe_errors(df)

    if not isinstance(remove, bool):
        raise TypeError("Parameter 'remove' must be a boolean")

    single_value_features = []

    if feature_list is None:
        feature_list = df.columns.tolist()

    # If ignore_cols is specified, add it to feature_list
    if ignore_cols:
        feature_list = [feature for feature in feature_list if feature != ignore_cols]

    feature_list = [feature for feature in feature_list if feature in df.columns]

    for feature in feature_list:
        if df[feature].nunique() == 1:
            single_value_features.append(feature)

    if remove:
        return df.drop(single_value_features, axis=1)
    else:
        return single_value_features

def plot_num_col(df, num_feature_list=None, figsize=None, column_size = None):
    # Plotting histograms of numerical features in the dataset

    if figsize is None:
        rows = 5*column_size
        columns = 7 - column_size
        plt.figure(figsize=(rows, columns))
    else:
        plt.figure(figsize=figsize)

    rows = math.ceil(len(num_feature_list)/column_size)
    for i, column in enumerate(num_feature_list):
        plt.subplot(rows, column_size, i+1)
        sns.histplot(df[column], kde=True)
        plt.title(f'Histogram of {column}')

    plt.tight_layout()
    plt.show()

def find_duplicate_col(df):
    columns = df.columns.tolist()
    duplicate_cols = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_i = columns[i]
            col_j = columns[j]
            if (df[col_i] == df[col_j]).all():
                duplicate_cols.append(col_i, "&", col_j)

    return duplicate_cols
