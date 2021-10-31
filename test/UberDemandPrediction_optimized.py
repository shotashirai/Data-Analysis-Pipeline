# coding: utf-8

""" 
Code: UberDemandPrediction.py
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
import os, sys
sys.path.append(os.getcwd())
import glob
import pandas as pd
import time

import numpy as np

from datetime import date, datetime
from functools import reduce
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
import lightgbm as lgb
from boruta import BorutaPy

# My data pipeline
from my_pipeline.load import load_file
from my_pipeline.dataprofile import data_profile
from my_pipeline.datacleaning import *
from my_pipeline.model import TimeSeries_XGBoost
from my_pipeline.save import save_prediction_csv, save_score_csv
from my_pipeline.utils import *
from my_pipeline.feat_engineer import gen_lagdata

start = time.time()
###################################################################################
# Import data 
###################################################################################
# Data is stored in the "data" directory
dir_name_Uber = 'data/uber-tlc-foil-response/uber-trip-data'
dir_name_Weather = 'data/weather'

#--------------- New code --------------------------------------------------------#
# In the old code, 'load_file' function in my_pipeline was used, but in the new version,
# read_scv() in the original pandas library is used due to the flexibility of coding.
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

df_uber['Borough'] = df_uber['locationID'].map(borough_ref.set_index('LocationID')['Borough'])
df_uber = df_uber.drop('locationID', axis=1)
df_uber = df_uber.rename(columns={'Pickup_date': 'datetime'}) # use the same datetime name as weather data

# delete borough_ref object
del borough_ref

df_uber.info(memory_usage='deep')
print(df_uber.memory_usage())
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

df_weather.info(memory_usage='deep')
print(df_weather.memory_usage())
#---------------------------------------------------------------------------------# 



# +=+= OLD CODE +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+= # OLD CODE - START
# Uber data
# uber_raw_janjun15 = load_file(dir_name_Uber, 'uber-raw-data-janjune-15.csv')
# taxi_zone = load_file(dir_name_Uber, 'taxi-zone-lookup.csv')

# # Weather data --------------------------------------------------------------------
# city_attrib = load_file(dir_name_Weather, 'city_attributes.csv')
# humidity = load_file(dir_name_Weather, 'humidity.csv')
# pressure = load_file(dir_name_Weather, 'pressure.csv')
# temperature = load_file(dir_name_Weather, 'temperature.csv')
# weather_description = load_file(dir_name_Weather, 'weather_description.csv')
# wind_direction = load_file(dir_name_Weather, 'wind_direction.csv')
# wind_speed = load_file(dir_name_Weather, 'wind_speed.csv')
# +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+ # OLD CODE - END


# +=+= OLD CODE +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+= # OLD CODE - START
# ###################################################################################
# # Preprocessing for analysis
# ###################################################################################

# # Uber data -----------------------------------------------------------------------
# # Borough
# uber_raw_janjun15['borough'] = uber_raw_janjun15['locationID'].map(taxi_zone.set_index('LocationID')['Borough'])
# uber_raw_janjun15 = datatime_format(uber_raw_janjun15, 'Pickup_date')
# print_timeperiod(uber_raw_janjun15, 'datetime')
# # Weather data --------------------------------------------------------------------
# # Extract weather in NY and merge weather data
# # merged weather data
# dataframes = [
#     humidity[['datetime','New York']]
#     , pressure[['datetime','New York']]
#     , temperature[['datetime','New York']]
#     , weather_description[['datetime','New York']]
#     , wind_direction[['datetime','New York']]
#     , wind_speed[['datetime','New York']]
# ]
# # merged data frame for weather data in NY
# weather_NY = reduce(lambda left, right: pd.merge(left, right , on='datetime',
#                    how='outer'), dataframes)

# weather_NY.columns = ['datetime', 'humidity', 'pressure', 'temperature'
#                     , 'weather_description', 'wind_direction', 'wind_speed']
# weather_NY.name = 'weather_NY'

# weather_NY = datatime_format(weather_NY, 'datetime', get_new_cols=False)
# print_timeperiod(weather_NY, 'datetime')
# # delete the unnecessary variables
# del humidity, pressure, temperature, weather_description, wind_direction, wind_speed

# # Weather data (01/01/2015 - 31/06/2015)
# weather_NY_15 = weather_NY.loc[(weather_NY['datetime'] < '2015-07') 
#                                 & (weather_NY['datetime'] > '2015')]
# weather_NY_15.name = 'weather_NY_15'
# del weather_NY
# +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+ # OLD CODE - END

elapsed_time = time.time() - start
print("Import Data - Preprocessing - elapsed_time:{0}".format(elapsed_time) + "[sec]")

start = time.time()
# ###################################################################################
# # Data Profile
# ###################################################################################
data_profile(df_uber)
data_profile(df_weather.reset_index())

elapsed_time = time.time() - start
print("Data Profile - elapsed_time:{0}".format(elapsed_time) + "[sec]")


start = time.time()
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

# add day of week(0:Sun, 1:Mon,...,6:Fri)
# df_uber_hourly['day_of_week'] = df_uber_hourly['datetime'].dt.dayofweek.astype('int8')
df_uber_hourly['day_of_week'] = df_uber_hourly['datetime'].dt.strftime('%a').astype('category')

df_model = df_uber_hourly.set_index('datetime').join(df_weather)


del df_uber, df_uber_hourly, df_weather

# Drop columns that are not used for modelling 
drop_cols = ['Staten Island', 'Unknown', 'EWR', 'Bronx']
df_model.drop(drop_cols, axis=1, inplace=True)

# Check memory usage ############################
# df_model.info(memory_usage='deep')
# data_profile(df_model.reset_index())
# ###############################################
elapsed_time = time.time() - start
print("Preprocessing - elapsed_time:{0}".format(elapsed_time) + "[sec]")


start = time.time()
# ###################################################################################
# # Feature Engineering 
# ###################################################################################

# Get dummy variables (one-hot label) for weather description -----------------------
df_model = pd.get_dummies(df_model)

# print(df_model.columns)
# >  ['Brooklyn', 'Manhattan', 'Queens', 'Total_rides',
#        'temperature', 'wind_direction', 'pressure', 'wind_speed', 'humidity',
#        'day_of_week_Fri', 'day_of_week_Mon', 'day_of_week_Sat',
#        'day_of_week_Sun', 'day_of_week_Thu', 'day_of_week_Tue',
#        'day_of_week_Wed', 'weather_description_broken clouds',
#        'weather_description_drizzle', 'weather_description_few clouds',
#        'weather_description_fog', 'weather_description_haze',
#        'weather_description_heavy intensity rain',
#        'weather_description_light intensity drizzle',
#        'weather_description_light rain', 'weather_description_light snow',
#        'weather_description_mist', 'weather_description_moderate rain',
#        'weather_description_overcast clouds',
#        'weather_description_proximity thunderstorm',
#        'weather_description_scattered clouds',
#        'weather_description_sky is clear', 'weather_description_snow',
#        'weather_description_thunderstorm',
#        'weather_description_thunderstorm with light rain',
#        'weather_description_very heavy rain']


# Get lag features ----------------------------------------------------------------

# Columns to generate lag features
lag_cols = df_model.columns.to_list()
# lags (1hr, 2hr, 3hr, 6hr, 12hr, 1day, 1week)
lags = [1, 2, 3, 6, 12, 24, 168]
# Get lag feature
df_model = gen_lagdata(df_model, lag_cols, lags)


# Standardize the variables ---------------------------------------------------------
# In this code, XGBoost is used for modelling and it does not need standardization of 
# features. Here, to compare with the old code, revised code is written, but in the 
# new version of the code, this part is commeted out.
from sklearn.preprocessing import StandardScaler
sc =  StandardScaler()
df_model_std = sc.fit_transform(df_model)


# +=+= OLD CODE +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+= # OLD CODE - START
# # extract date and time (hour)
# uber_raw_janjun15['datehour'] = uber_raw_janjun15['datetime'].dt.floor('1h')
# weather_NY_15['datehour'] = weather_NY_15['datetime'].dt.floor('1h')

# # hourly rides from all borough
# hourly_total_rides = uber_raw_janjun15[['datehour','datetime']]\
#                     .groupby('datehour')\
#                     .count()\
#                     .reset_index()
# hourly_total_rides.columns  = ['datehour', 'count']

# # hourly rides from all borough
# hourly_rides_borough = uber_raw_janjun15[['datehour','datetime','borough']]\
#                         .groupby(['datehour','borough'])\
#                         .count()\
#                         .reset_index()
# hourly_rides_borough.columns  = ['datehour','borough', 'count']
# # Merge Uber data and Weather data-------------------------------------------------
# df_hourly_rides = pd.merge(hourly_total_rides, weather_NY_15, on='datehour')
# df_hourly_rides_borough = pd.merge(hourly_rides_borough, weather_NY_15, on='datehour')
# # add day of week(0:Sun, 1:Mon,...,6)
# df_hourly_rides['day_of_week'] = df_hourly_rides['datetime'].dt.strftime('%w')
# df_hourly_rides_borough['day_of_week'] = df_hourly_rides_borough['datetime'].dt.strftime('%w')


# ###################################################################################
# # Feature Engineering 
# ###################################################################################
# df_model_rides = df_hourly_rides.copy()
# df_model_rides = df_model_rides.drop(['datehour','day_of_week'], axis=1)

# +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+ # OLD CODE - END

# # Standardize the variables
# col_names = ['humidity', 'temperature', 'pressure', 'wind_direction', 'wind_speed']
# df_model_rides = standard_vars(df_model_rides, col_names)
# df_model_rides['day_of_week'] = df_model_rides['datetime'].dt.strftime('%a')
# # Get one-hot label data for weather description 
# df_model_rides = pd.get_dummies(df_model_rides)


#### Deleted : this part is included in df_rides_hourly / df_model ###################################
# +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+ # OLD CODE - Start
# # Dataframe by borough
# df_hour_borough = df_hourly_rides_borough[['datetime','datehour','day_of_week','count','borough']]
# df_hour_borough['hour'] = df_hour_borough['datetime'].dt.strftime('%H')
# borough_list_model = ['Manhattan', 'Brooklyn', 'Queens']
# df_borough_model = df_hour_borough[['datetime', 'count', 'borough']]
# for borough in borough_list_model:
#     df_borough = df_borough_model.loc[df_borough_model['borough']==borough]
#     df_borough=df_borough.drop(['borough'], axis=1)
#     df_borough.columns = ['datetime']+['rides_'+borough]
#     df_model_rides = pd.merge(df_model_rides, df_borough, on='datetime', how='left')
# +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+ # OLD CODE - END
#### Deleted : this part is included in df_rides_hourly / df_model ####################################

# # Generate lag feature
# lag_variables = [
#         'count'
#         , 'rides_Manhattan', 'rides_Brooklyn', 'rides_Queens'
#         ,'humidity', 'pressure', 'temperature',
#         'wind_direction', 'wind_speed', 'weather_description_broken clouds',
#         'weather_description_drizzle', 'weather_description_few clouds',
#         'weather_description_fog', 'weather_description_haze',
#         'weather_description_heavy intensity rain',
#         'weather_description_light intensity drizzle',
#         'weather_description_light rain', 'weather_description_light snow',
#         'weather_description_mist', 'weather_description_moderate rain',
#         'weather_description_overcast clouds',
#         'weather_description_proximity thunderstorm',
#         'weather_description_scattered clouds',
#         'weather_description_sky is clear', 'weather_description_snow',
#         'weather_description_thunderstorm',
#         'weather_description_thunderstorm with light rain',
#         'weather_description_very heavy rain',
#         'day_of_week_Fri',
#         'day_of_week_Mon', 'day_of_week_Sat', 'day_of_week_Sun',
#         'day_of_week_Thu', 'day_of_week_Tue', 'day_of_week_Wed'
#         ]

# # lags (1hr, 2hr, 3hr, 6hr, 12hr, 1day, 1week)
# lags = [1, 2, 3, 6, 12, 24, 168]

# df_model = gen_lagdata(df_model_rides, lag_variables, lags)
elapsed_time = time.time() - start
print("Feature Enginnering - elapsed_time:{0}".format(elapsed_time) + "[sec]")

df_model.info(memory_usage='deep')
print(df_model.memory_usage())

start = time.time()
# ###################################################################################
# # Model
# ###################################################################################

# print('Time period of Train data')
# print('Min time: %s' % df_model.reset_index()['datetime'].min())
# print('Man time: %s' % df_model.reset_index()['datetime'].max())

# > Time period of Train data
# > Min time: 2015-01-08 00:00:00
# > Min time: 2015-06-30 23:00:00

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
    model_base = XGBR(seed=rand_seed)
    results_base = TimeSeries_XGBoost(model_base, df_train_model, df_test_model, borough)

    print('--- Tuned Model -------------------------------------')
    # model with different paramters
    model_tuned = XGBR(seed=rand_seed
                    , max_depth=20
                    , n_estimators=1000
                    )
    results_tuned = TimeSeries_XGBoost(model_tuned, df_train_model, df_test_model, borough)

    print('--- Tuned Model + Feature selection------------------')

    model_tuned = XGBR(seed=rand_seed
                    , max_depth=20
                    , n_estimators=1000
                    )

    # Feature Selection
    y_train = df_train[borough]
    X_train = df_train.drop(borough, axis=1)

    ########### XGBoost ################################################################
    # start = time.time()
    # # Test1: XGBoost
    # model_xgb = XGBR(seed=rand_seed, max_depth=20, n_estimators=1000)
    # feat_selector_xgb = BorutaPy(model_xgb, n_estimators='auto', verbose=0, random_state=rand_seed)
    # feat_selector_xgb.fit(X_train.values, y_train.values)
    # selected_feat = X_train.iloc[:,feat_selector_xgb.support_].columns.to_list() + [borough]
    # # Extract selected features
    # df_train_selected = df_train[selected_feat]
    # df_test_selected = df_test[selected_feat]
    # results_xgb = TimeSeries_XGBoost(model_tuned, df_train_selected, df_test_selected, borough)
    
    # elapsed_time = time.time() - start
    # print("XGBoost - elapsed_time:{0}".format(elapsed_time) + "[sec]")
    #####################################################################################


    ########### Random Forest ###########################################################
    # model_tuned = XGBR(seed=rand_seed
    #                 , max_depth=20
    #                 , n_estimators=1000
    #                 )
    # start = time.time()
    # # Test2: RandomForestRegressor
    # rfr = RFR(max_depth=5, n_jobs=-1)
    # feat_selector_rfc = BorutaPy(rfr, n_estimators='auto', verbose=0, random_state=rand_seed)
    # feat_selector_rfc.fit(X_train.values, y_train.values)
    # selected_feat = X_train.iloc[:,feat_selector_rfc.support_].columns.to_list() + [borough]
    # # Extract selected features
    # df_train_selected = df_train[selected_feat]
    # df_test_selected = df_test[selected_feat]
    # results_rfr = TimeSeries_XGBoost(model_tuned, df_train_selected, df_test_selected, borough)
    
    # elapsed_time = time.time() - start
    # print("RandomForest - elapsed_time:{0}".format(elapsed_time) + "[sec]")
    #####################################################################################
    # model_tuned = XGBR(seed=rand_seed
    #                 , max_depth=20
    #                 , n_estimators=1000
    #                 )
    # start = time.time()
    # Test 3: LightGBM
    model_lgb = lgb.LGBMRegressor(num_iterations=100)
    feat_selector_lgb = BorutaPy(model_lgb, n_estimators='auto', verbose=0, random_state=rand_seed)
    feat_selector_lgb.fit(X_train.values, y_train.values)
    selected_feat = X_train.iloc[:,feat_selector_lgb.support_].columns.to_list() + [borough]
    # Extract selected features
    df_train_selected = df_train[selected_feat]
    df_test_selected = df_test[selected_feat]
    results_lgb = TimeSeries_XGBoost(model_tuned, df_train_selected, df_test_selected, borough)
    
elapsed_time = time.time() - start
print("Modelling - elapsed_time:{0}".format(elapsed_time) + "[sec]")


# # Split the data into test data and Validation data
# # Length of test data (days)
# test_length = 30

# df_train = df_model[df_model['time_block_num'] <= (df_model['time_block_num'].max()-24*test_length)]
# df_test = df_model[df_model['time_block_num'] > (df_model['time_block_num'].max()-24*test_length)]

# print('Time period of Train data')
# print('Min time: %s' % df_train['datetime'].min())
# print('Min time: %s' % df_train['datetime'].max())
# print('\nTime period of Test data')
# print('Min time: %s' % df_test['datetime'].min())
# print('Min time: %s' % df_test['datetime'].max())

# # Keep the datetime info with 'time_block_num'
# time_block = df_model[['datetime', 'time_block_num']]
# df_model = df_model.drop('datetime', axis=1)
# df_train = df_train.drop('datetime', axis=1)
# df_test = df_test.drop('datetime', axis=1)

# # Borough list 
# target_list = ['count', 'rides_Manhattan', 'rides_Brooklyn', 'rides_Queens']
# # Random seed
# rand_seed = 42
# # Baseline Model (Without parameter tuning)
# model = XGBR(seed=rand_seed)
# print('=== Base Model =====================================')
# results1 = prediction_XGBoost(df_train, df_test, model, target_list)
# # model with different paramters
# model2 = XGBR(
#     max_depth=20
#     , n_estimators=1000
#     , seed=rand_seed)
# print('=== Model (tuned parameters) =======================')
# results2 = prediction_XGBoost(df_train, df_test, model2, target_list)
# # model2 + feature selection
# print('=== Model (tuned parameters) + Feature selection ===')
# results3 = prediction_XGBoost(df_train, df_test, model2, target_list, feature_select=True)
# ###################################################################################
# # Deploy
# ###################################################################################
# # TODO need to develop a function to save the results.
# # The main issue here is a format of the result.It needs to use the same format in different analysis.
