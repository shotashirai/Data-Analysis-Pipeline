# coding: utf-8

""" 
Code: UberDemandPrediction.py (Optimized Version)
Author: Shota Shirai
Input: uber-raw-data-janjune-15.csv, weather data in NY
Output: 
Source data: 
    - Uber trip data: https://github.com/fivethirtyeight/uber-tlc-foil-response
    - Historical weather in NY: https://www.kaggle.com/selfishgene/historical-hourly-weather-data?select=humidity.csv

Required external codes/modules: provided by 'my_pipeline'

This code predicts Uber Demand in New York based on the analyses on 
the Jupyter Notebook (Uber-Demand-Forecasting_EDA_Model.ipynb) in 'EDA_jupyter_notebook' directory. 

Note: Tested Python version 3.8.10
"""

###################################################################################
# Import libraries 
###################################################################################

# General libraries
import os, sys
sys.path.append(os.pardir)
import glob
import pandas as pd
# import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Model
from xgboost import XGBRegressor as XGBR
import lightgbm as lgb
from boruta import BorutaPy

# My pipeline
from my_pipeline.dataprofile import data_profile
from my_pipeline.feat_engineer import gen_lagdata
from my_pipeline.model import TimeSeries_XGBoost


###################################################################################
# Import data 
###################################################################################
# Data is stored in the "data" directory
dir_name_Uber = 'data/uber-tlc-foil-response/uber-trip-data'
dir_name_Weather = 'data/weather'

# Uber data ===================================================================== #
df_uber = pd.read_csv('/'.join([dir_name_Uber, 'uber-raw-data-janjune-15.csv'])
                    , usecols=['Pickup_date', 'locationID']
                    , dtype={'locationID': 'int16'}
                    , parse_dates=['Pickup_date']
                    )

# Reference for Borough
borough_ref = taxi_zone = pd.read_csv('data/uber-tlc-foil-response/uber-trip-data/taxi-zone-lookup.csv'
                                    , usecols=['LocationID', 'Borough']
                                    , dtype={'LocationID': 'int16', 'Borough':'category'}
                                    )
# Map borough based on location ID
df_uber['Borough'] = df_uber['locationID'].map(borough_ref.set_index('LocationID')['Borough'])
df_uber = df_uber.drop('locationID', axis=1)

# Rename datetime column ('Pickup_date' --> 'datetime')
df_uber = df_uber.rename(columns={'Pickup_date': 'datetime'}) # use the same datetime name as weather data

# delete borough_ref object
del borough_ref

# Weather data =================================================================== #

# get a file list of the weather data
weather_filelist = glob.glob('/'.join([dir_name_Weather,'*.csv']))

# Get a list of the weather data
df_list = [pd.read_csv(filename
                        , usecols=[0, 28] #0: datetime, 28: New York 
                        , names=('datetime', os.path.splitext(os.path.basename(filename))[0]) # Rename the columns
                        , header=0
                        , parse_dates=['datetime']
                        , index_col='datetime'
                        ) for filename in weather_filelist]

# Concatenate the files in the list
df_weather = pd.concat(df_list, axis=1)

# Extract data with the same time period of the uber data
df_weather = df_weather.loc['2015-01-01':'2015-06-30']

# Convert the data types (No NaN values for this time period and range of values were already checked)
df_weather = df_weather.astype({
    'temperature': 'float32'
    , 'humidity': 'int8'
    , 'pressure': 'int16'
    , 'weather_description': 'category'
    , 'wind_direction': 'int16'
    , 'wind_speed': 'int8'
})

# ###################################################################################
# # Data Profile
# ###################################################################################

data_profile(df_uber)
print('...')
data_profile(df_weather.reset_index())

# ###################################################################################
# # Preprocessing
# ###################################################################################

# Get Hourly Uber rides 
df_uber['Pickup'] = 1
df_uber_hourly = df_uber.groupby([pd.Grouper(key='datetime', freq='1H'), 'Borough'])
df_uber_hourly = df_uber_hourly['Pickup'].count().astype('int16').unstack()

# Add column for total number of the hourly rides
df_uber_hourly['Total_rides'] = df_uber[['datetime', 'Pickup']]\
                                .groupby([pd.Grouper(key='datetime', freq='1H')])\
                                .count().astype('int16')

# reset index
df_uber_hourly = df_uber_hourly.reset_index()
df_uber_hourly.columns.name = ''

# add day of week(Sun, Mon,..., Fri)
df_uber_hourly['day_of_week'] = df_uber_hourly['datetime'].dt.strftime('%a').astype('category')

# Concatenate the uber data and the weather data
df_model = df_uber_hourly.set_index('datetime').join(df_weather)

# Delete unnecessary objects
del df_uber, df_uber_hourly, df_weather

# Check memory usage ############################
# df_model.info(memory_usage='deep')
# data_profile(df_model.reset_index())
# ###############################################

###################################################################################
# Feature Engineering 
###################################################################################

# Get dummy variables (one-hot label) for weather description ---------------------
df_model = pd.get_dummies(df_model)

# Get lag features ----------------------------------------------------------------

# Columns to generate lag features
lag_cols = df_model.columns.to_list()

# lags (1hr, 2hr, 3hr, 6hr, 12hr, 1day, 1week)
lags = [1, 2, 3, 6, 12, 24, 168]

# Get lag feature
df_model = gen_lagdata(df_model, lag_cols, lags)

###################################################################################
# Model
###################################################################################

# Split data into train data and test data
# For test data, data in June is used.
df_test = df_model.loc['2015-06-01':'2015-06-30']
df_train = df_model.loc['2015-01-01':'2015-05-31']

# Borough list for forecasting
Borough_list = ['Total_rides', 'Manhattan', 'Brooklyn', 'Queens']

# Random seed
rand_seed = 42

for borough in Borough_list:
    
    print(''.join(['====== Uber Demand Forecasting (', borough, ') ======']))
    # columns for dropping from df_model
    drop_cols = Borough_list.copy()
    drop_cols.remove(borough)
    # dataframe for modelling
    df_train_model = df_train.drop(drop_cols, axis=1)
    df_test_model = df_test.drop(drop_cols, axis=1)

    print('--- Base Model --------------------------------------')
    
    # Base model
    model_base = XGBR(seed=rand_seed)
    
    # Forecasting using XGBoost
    results_base = TimeSeries_XGBoost(model_base, df_train_model, df_test_model, borough)

    print('--- Tuned Model -------------------------------------')
    # model with different paramters
    model_tuned = XGBR(seed=rand_seed
                    , max_depth=20
                    , n_estimators=1000
                    )

    # Forecasting using XGBoost
    results_tuned = TimeSeries_XGBoost(model_tuned, df_train_model, df_test_model, borough)

    print('--- Tuned Model + Feature selection------------------')
    
    # XGBoost regressor model
    model_tuned = XGBR(seed=rand_seed
                    , max_depth=20
                    , n_estimators=1000
                    )

    # Feature Selection
    y_train = df_train[borough]
    X_train = df_train.drop(borough, axis=1)

    # Feature selection with BorutaPy + LightGBM ----------------------------------------------

    # lightGBM
    model_lgb = lgb.LGBMRegressor(num_iterations=100)

    # Feature Selector
    feat_selector_lgb = BorutaPy(model_lgb, n_estimators='auto', verbose=0, random_state=rand_seed)
    feat_selector_lgb.fit(X_train.values, y_train.values)

    # Get selected features
    selected_feat = X_train.iloc[:,feat_selector_lgb.support_].columns.to_list() + [borough]

    # Extract selected features
    df_train_selected = df_train[selected_feat]
    df_test_selected = df_test[selected_feat]

    # Forecasting using XGBoost
    results_lgb = TimeSeries_XGBoost(model_tuned, df_train_selected, df_test_selected, borough)
