# coding: utf-8

""" 
Code: TitanicSurvivalPrediction.py
Author: Shota Shirai
Input: train.csv, test.csv
Output: survival_prediction_(model name).csv

Source data: 
    - Uber trip data: https://github.com/fivethirtyeight/uber-tlc-foil-response
    - Histroical weather in NY: https://www.kaggle.com/selfishgene/historical-hourly-weather-data?select=humidity.csv

Required external codes/moduls: provided by 'my_pipeline'

This code processes the titanic passenger data provided by Kaggle based on the analyses 
on the Jupyter Notebook (TitanicSurvivalPrediction_EDA_Model.ipynb) 
in 'EDA_jupyter_notebook' directory. 

Note: Tested Python version 3.8.10
"""
###################################################################################
# Import libraries 
###################################################################################
import os, sys
sys.path.append(os.pardir) # to import files in the parent directory
import numpy as np
import pandas as pd
from itertools import compress
from datetime import date, datetime
from functools import reduce
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor as XGBR

# My data pipeline
from my_pipeline.load import load_file
from my_pipeline.dataprofile import data_profile
from my_pipeline.datacleaning import *
from my_pipeline.model import * 
from my_pipeline.save import save_prediction_csv, save_score_csv
from my_pipeline.utils import *
from my_pipeline.feat_engineer import *
###################################################################################
# Import data 
###################################################################################
# Data is sotred in the "data" directory
dir_name_Uber = 'data/uber-tlc-foil-response/uber-trip-data'
dir_name_Weather = 'data/weather'

# Uber data
uber_raw_janjun15 = load_file(dir_name_Uber, 'uber-raw-data-janjune-15.csv')
taxi_zone = load_file(dir_name_Uber, 'taxi-zone-lookup.csv')

# Weather data --------------------------------------------------------------------
city_attrib = load_file(dir_name_Weather, 'city_attributes.csv')
humidity = load_file(dir_name_Weather, 'humidity.csv')
pressure = load_file(dir_name_Weather, 'pressure.csv')
temperature = load_file(dir_name_Weather, 'temperature.csv')
weather_description = load_file(dir_name_Weather, 'weather_description.csv')
wind_direction = load_file(dir_name_Weather, 'wind_direction.csv')
wind_speed = load_file(dir_name_Weather, 'wind_speed.csv')

###################################################################################
# Preprocessing for analysis
###################################################################################

# Uber data -----------------------------------------------------------------------
# Borough
uber_raw_janjun15['borough'] = uber_raw_janjun15['locationID'].map(taxi_zone.set_index('LocationID')['Borough'])
uber_raw_janjun15 = datatime_format(uber_raw_janjun15, 'Pickup_date')
print_timeperiod(uber_raw_janjun15, 'datetime')
# Weather data --------------------------------------------------------------------
# Extract weather in NY and merge weather data
# merged weather data
dataframes = [
    humidity[['datetime','New York']]
    , pressure[['datetime','New York']]
    , temperature[['datetime','New York']]
    , weather_description[['datetime','New York']]
    , wind_direction[['datetime','New York']]
    , wind_speed[['datetime','New York']]
]
# merged data frame for weather data in NY
weather_NY = reduce(lambda left, right: pd.merge(left, right , on='datetime',
                   how='outer'), dataframes)

weather_NY.columns = ['datetime', 'humidity', 'pressure', 'temperature'
                    , 'weather_description', 'wind_direction', 'wind_speed']
weather_NY.name = 'weather_NY'

weather_NY = datatime_format(weather_NY, 'datetime', get_new_cols=False)
print_timeperiod(weather_NY, 'datetime')
# delete the unnecessary variables
del humidity, pressure, temperature, weather_description, wind_direction, wind_speed

# Weather data (01/01/2015 - 31/06/2015)
weather_NY_15 = weather_NY.loc[(weather_NY['datetime'] < '2015-07') 
                                & (weather_NY['datetime'] > '2015')]
weather_NY_15.name = 'weather_NY_15'
del weather_NY
###################################################################################
# Data Profile
###################################################################################
data_profile(uber_raw_janjun15)
data_profile(weather_NY_15)

###################################################################################
# Preprocessing
###################################################################################

# extract date and time (hour)
uber_raw_janjun15['datehour'] = uber_raw_janjun15['datetime'].dt.floor('1h')
weather_NY_15['datehour'] = weather_NY_15['datetime'].dt.floor('1h')

# hourly rides from all borough
hourly_total_rides = uber_raw_janjun15[['datehour','datetime']]\
                    .groupby('datehour')\
                    .count()\
                    .reset_index()
hourly_total_rides.columns  = ['datehour', 'count']

# hourly rides from all borough
hourly_rides_borough = uber_raw_janjun15[['datehour','datetime','borough']]\
                        .groupby(['datehour','borough'])\
                        .count()\
                        .reset_index()
hourly_rides_borough.columns  = ['datehour','borough', 'count']
# Merge Uber data and Weather data-------------------------------------------------
df_hourly_rides = pd.merge(hourly_total_rides, weather_NY_15, on='datehour')
df_hourly_rides_borough = pd.merge(hourly_rides_borough, weather_NY_15, on='datehour')
# add day of week(0:Sun, 1:Mon,...,6)
df_hourly_rides['day_of_week'] = df_hourly_rides['datetime'].dt.strftime('%w')
df_hourly_rides_borough['day_of_week'] = df_hourly_rides_borough['datetime'].dt.strftime('%w')


###################################################################################
# Feature Engineering 
###################################################################################
df_model_rides = df_hourly_rides.copy()
df_model_rides = df_model_rides.drop(['datehour','day_of_week'], axis=1)
# Standardize the variables
col_names = ['humidity', 'temperature', 'pressure', 'wind_direction', 'wind_speed']
df_model_rides = standard_vars(df_model_rides, col_names)
df_model_rides['day_of_week'] = df_model_rides['datetime'].dt.strftime('%a')
# Get one-hot label data for weather description 
df_model_rides = pd.get_dummies(df_model_rides)

# Dataframe by borough
df_hour_borough = df_hourly_rides_borough[['datetime','datehour','day_of_week','count','borough']]
df_hour_borough['hour'] = df_hour_borough['datetime'].dt.strftime('%H')

borough_list_model = ['Manhattan', 'Brooklyn', 'Queens']
df_borough_model = df_hour_borough[['datetime', 'count', 'borough']]
for borough in borough_list_model:
    df_borough = df_borough_model.loc[df_borough_model['borough']==borough]
    df_borough=df_borough.drop(['borough'], axis=1)
    df_borough.columns = ['datetime']+['rides_'+borough]
    df_model_rides = pd.merge(df_model_rides, df_borough, on='datetime', how='left')

# Generate lag feature
lag_variables = [
        'count'
        , 'rides_Manhattan', 'rides_Brooklyn', 'rides_Queens'
        ,'humidity', 'pressure', 'temperature',
        'wind_direction', 'wind_speed', 'weather_description_broken clouds',
        'weather_description_drizzle', 'weather_description_few clouds',
        'weather_description_fog', 'weather_description_haze',
        'weather_description_heavy intensity rain',
        'weather_description_light intensity drizzle',
        'weather_description_light rain', 'weather_description_light snow',
        'weather_description_mist', 'weather_description_moderate rain',
        'weather_description_overcast clouds',
        'weather_description_proximity thunderstorm',
        'weather_description_scattered clouds',
        'weather_description_sky is clear', 'weather_description_snow',
        'weather_description_thunderstorm',
        'weather_description_thunderstorm with light rain',
        'weather_description_very heavy rain',
        'day_of_week_Fri',
        'day_of_week_Mon', 'day_of_week_Sat', 'day_of_week_Sun',
        'day_of_week_Thu', 'day_of_week_Tue', 'day_of_week_Wed'
        ]

# lags (1hr, 2hr, 3hr, 6hr, 12hr, 1day, 1week)
lags = [1, 2, 3, 6, 12, 24, 168]

df_model = gen_lagdata(df_model_rides, lag_variables, lags)

###################################################################################
# Model
###################################################################################
# TODO make a function to split time-series data into training and test dataset
# Split the data into test data and Validation data
# Length of test data (days)
test_length = 30

df_train = df_model[df_model['time_block_num'] <= (df_model['time_block_num'].max()-24*test_length)]
df_test = df_model[df_model['time_block_num'] > (df_model['time_block_num'].max()-24*test_length)]

print('Time period of Train data')
print('Min time: %s' % df_train['datetime'].min())
print('Min time: %s' % df_train['datetime'].max())
print('\nTime period of Test data')
print('Min time: %s' % df_test['datetime'].min())
print('Min time: %s' % df_test['datetime'].max())

# Keep the datetime info with 'time_block_num'
time_block = df_model[['datetime', 'time_block_num']]
df_model = df_model.drop('datetime', axis=1)
df_train = df_train.drop('datetime', axis=1)
df_test = df_test.drop('datetime', axis=1)

# Borough list 
target_list = ['count', 'rides_Manhattan', 'rides_Brooklyn', 'rides_Queens']
# Random seed
rand_seed = 42
# Baseline Model (Without parameter tuning)
model = XGBR(seed=rand_seed)
print('=== Base Model =====================================')
results1 = prediction_XGBoost(df_train, df_test, model, target_list)
# model with different paramters
model2 = XGBR(
    max_depth=20
    , n_estimators=1000
    , seed=rand_seed)
print('=== Model (tuned parameters) =======================')
results2 = prediction_XGBoost(df_train, df_test, model2, target_list)
# model2 + feature selection
print('=== Model (tuned parameters) + Feature selection ===')
results3 = prediction_XGBoost(df_train, df_test, model2, target_list, feature_select=True)
###################################################################################
# Deploy
###################################################################################
# TODO need to develop a function to save the results.
# The main issue here is a format of the result.It needs to use the same format in different analysis.
