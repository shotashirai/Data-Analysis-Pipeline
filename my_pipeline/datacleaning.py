# coding:utf-8

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR

    
def repalce_nan_str(df, col_names, new_strs):
    """ Replace NaN values with string
    
    Parameters
    ----------
    df : Original dataframe
    col_names (list): column names to be checked  
    new_strs (list): new strings for replacement

    Returns
    -------
    df: nan is replaced by new string
    """
    # Repalce nan values in each column 
    for col, rep_value in zip(col_names, new_strs):
        df[col] = df[col].fillna(rep_value)

    return df


def replace_nan_rfr(df, df_model, col_name):
    """ Replace NaN values with values predicted by Random Forest Regressor
    
    Parameters
    ----------
    df : Original dataframe
    df_model: data used for random forest model
    col_name: column name of the column for replacement

    Returns
    -------
    df: nan is replaced by new predicted valeus
    """

    # One-hot encoding
    df_model = pd.get_dummies(df_model)

    # Split df_model into training data and test data
    null_df = df_model[df_model[col_name].isnull()] # test data
    notnull_df = df_model[df_model[col_name].notnull()] # training data
    
    # data for training 
    y = notnull_df[col_name] # label
    X = notnull_df.drop(col_name, axis=1) # variables

    # data for prediction
    X_pred = null_df.drop(col_name, axis=1)

    # Build a model using Random forest regressor 
    rfr = RFR(random_state=0, n_estimators=100, n_jobs=-1) 
    rfr.fit(X, y)

    predicted_values = rfr.predict(X_pred)
    df.loc[(df[col_name].isnull()), col_name] = predicted_values

    return df


