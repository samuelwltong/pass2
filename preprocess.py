#!/usr/bin/env python
# coding: utf-8

import sqlite3
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV

def get_data(db_name, tbl_name):
# Replace Nan with empty string for better presentation (note that in previous step, we've checked
    """ This function is used to extract data from data base file using sqlite3 library and
        return the data in dataframe format.
    db_name: name of the database file, with extension of the file. Dtype: str
    tbl_name: name of the table to extract data. Dtype: str
    """

    # Reset Dataframe Display Configuration
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')

    # Changing directory for program to be flexible, mainly for Task 2
    absolute_path = os.path.abspath(__file__)
    to_path = absolute_path[:-23]
    os.chdir(to_path)
    conn = sqlite3.connect("./data/" + db_name)
    df = pd.read_sql_query("select * from " + tbl_name,conn)
    conn.close()

    return df

def display_unique(df, drop_list=[]):
    """ This function is used to convert temperature from Fahrenheit to Celsius.
    df: dataframe to display unique values. Dtype: pandas.core.frame.DataFrame
    drop_list: a list of column names to exclude for the check. Dtype: list
    """

    # Reset Dataframe Display Configuration
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')

    # Defining tools to be used
    indices = []
    df1 = pd.DataFrame() 
    data = {}

    # Loop through columns and get unique values
    for col in df.drop(columns=drop_list).columns:
        if df[col].dtype == 'object':
            data[col] = df[col].unique()
        elif df[col].dtype in ['int64', 'float64']:
            data[col] = np.sort(df[col].unique())

    # Loop through mapping dictionary and add result to dataframe row by row
    for k in data.keys():
        df1 = pd.concat([df1, pd.DataFrame(data[k]).T], ignore_index=True)
        indices.append(k)

    # Re-define shape for dataframe
    df1.index = indices
    df1.columns = df1.columns + 1

    # Change configuration to display all columns in dataframe
    pd.options.display.max_columns = None

    return df1

def fahrenheit_to_celsius(temp):
    """ This function convert temperature from Fahrenheit to Celsius
    temp: temperature in Fahrenheit. Dtype: str
    """

    # Strip symbol and convert
    temp = temp.strip(" °F")
    celsius = (float(temp) - 32) * (5/9)
    return celsius

def convert_temperature(temp):
    """ This function convert temperature in string to Celsius in float
    temp: temperature. Dtype: str
    """

    # Strip symbol and convert
    if "°F" in temp:
        return fahrenheit_to_celsius(temp)
    elif "°C" in temp:
        temp = temp.strip(" °C")
        return float(temp)
    else:
        return temp

def dup_check(df, all_col, col_ = ""):
    """ This function is check for duplicates.
    df: dataframe to check. Dtype: pandas.core.frame.DataFrame
    all_col: boolean operator to decide whether to check duplicate in all columns. Dtype: bool
    col_: column name to check for duplicates. Dtype: str
    """
    # Reset Dataframe Display Configuration
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')

    # To check duplicates in all columns or only specific column
    if all_col:
        df_col = df.columns
        df_dup = df[df.duplicated(df_col, keep=False)]
    else:
        df_dup = df[df.duplicated(col_, keep=False)]
        df_dup = df_dup.sort_values (by= [col_], ascending=True)
        
    return df_dup

def data_wrangling(df, col_, method_):
    """ This function is used to perform feature engineering.
    df: dataframe to process. Dtype: pandas.core.frame.DataFrame
    col_: column name to process. Dtype: str
    method_: method of feature engineering. Dtype: str
    """

    if method_ == 'str_to_int':
        # replace string values with integers, e.g. replacing "1" & "0" with 1 & 0
        df[col_] = df[col_].astype(int)
    elif method_ == 'bool_to_binary':
        # replace "Yes" & "No" with 1 & 0
        df[col_] = df[col_].replace('Yes', 1, regex=True)
        df[col_] = df[col_].replace('No', 0, regex=True)
    elif method_ == 'upp_case':
        # convert string to upper case
        df[col_] = df[col_].str.upper()
    elif method_ == 'absolute':
        # convert negative values to positive values
        df[col_] = df[col_].abs()
        
    return df[col_]

def data_transform(df, col_, method_):
    """ This function is used to transform feature's distribution.
    df: dataframe to process. Dtype: pandas.core.frame.DataFrame
    col_: column name to process. Dtype: str
    method_: method of transformation. Dtype: str
    """

    if method_ == 'reci':
        df['new'] = 1/df[col_]
    elif method_ == 'boc_cox':
        df['new'], parameters = stats.boxcox(df[col_])
    return df['new']

def wrapper_method(df, numerical_cols, categorical_cols, target_cols, feature_cols, classifier_, scoring_):
    """ This function is used to perform the wrapper method of features selection.
    df: dataframe to process. Dtype: pandas.core.frame.DataFrame
    numerical_cols: numerical column name to process. Dtype: list or str
    categorical_cols: categorical column name to process. Dtype: list or str
    target_cols: target variable(s) column name to process. Dtype: list or str
    feature_cols: name of all features. Dtype: list or str
    classifier_: the type of classifier used. Dtype: varies with model
    scoring_: evaluation metric to be used. Dtype: str
    """

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df[feature_cols],
                                                        df[target_cols],
                                                        test_size = 0.2,
                                                        random_state = 42)

    # Create an instance of the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', MinMaxScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    # Preprocess the training data
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Create an instance of the RFECV class
    selector = RFECV(estimator = classifier_,
                                                cv = 5,
                                                scoring = scoring_)

    # Fit the RFECV to the preprocessed training data
    selector.fit(X_train_transformed, y_train)

    # Get the selected features
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.support_[i]]

    # Print the selected features
    print(selected_features)