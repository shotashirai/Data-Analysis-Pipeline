# coding: utf-8
import sys, os
sys.path.append(os.getcwd())
import pandas as pd
from my_pipeline.load import load_file

###################################################################################
# Import data 
###################################################################################
import glob

# Data is stored in the "data" directory
dir_name_Uber = 'data/uber-tlc-foil-response/uber-trip-data'
dir_name_Weather = 'data/weather'


#--------------- New code --------------------------------------------------------#

# Uber data
df_uber = pd.read_csv('/'.join([dir_name_Uber, 'uber-raw-data-janjune-15.csv'])
                    , usecols=['Pickup_date', 'locationID']
                    , dtype={'locationID': 'int16'})
# Reference for Borough
borough_ref = taxi_zone = pd.read_csv('data/uber-tlc-foil-response/uber-trip-data/taxi-zone-lookup.csv'
                                    , usecols=['LocationID', 'Borough']
                                    , dtype={'LocationID': 'int16', 'Borough':'category'}
                                    )


df_uber['Borough'] = df_uber['locationID'].map(borough_ref.set_index('LocationID')['Borough'])
df_uber = df_uber.drop('locationID', axis=1)

del borough_ref

### Print Memory usage ###
print('==Memory Usage: Uber data ==================')
df_uber.info(memory_usage='deep')
print(df_uber.memory_usage())
print('============================================')
##########################


# Weather data
weather_filelist = glob.glob('/'.join([dir_name_Weather,'*.csv']))

# Get a list of the weather data
df_list = [pd.read_csv(filename
                        , usecols=[0, 28] #0: datetime, 28: New York 
                        , names=('datetime', os.path.splitext(os.path.basename(filename))[0])
                        , header=0
                        , parse_dates=['datetime']
                        , index_col='datetime'
                        ) for filename in weather_filelist]
# Concatenate the files in the list
df_weather = pd.concat(df_list, axis=1)

# Extract data with the same time period of the uber data
df_weather = df_weather.loc['2015-01-01':'2015-06-30']

# Convert the data types (No NaN values for this time period)
df_weather = df_weather.astype({
    'temperature': 'float32'
    , 'humidity': 'int8'
    , 'pressure': 'int16'
    , 'weather_description': 'category'
    , 'wind_direction': 'int16'
    , 'wind_speed': 'int8'
})

### Print Memory usage ###
print('==Memory Usage: Weather data ===============')
df_weather.info(memory_usage='deep')
print(df_weather.memory_usage())
print('============================================')
##########################
#---------------------------------------------------------------------------------#




# Uber data
uber_raw_janjun15 = load_file(dir_name_Uber, 'uber-raw-data-janjune-15.csv')
taxi_zone = load_file(dir_name_Uber, 'taxi-zone-lookup.csv')

# Borough
uber_raw_janjun15['borough'] = uber_raw_janjun15['locationID']\
                            .map(taxi_zone.set_index('LocationID')['Borough'])

# convert string to timestamp object
uber_raw_janjun15['datetime'] = pd.to_datetime(uber_raw_janjun15['Pickup_date'])

print('==Memory Usage: Uber data (old) ============')
uber_raw_janjun15.info(memory_usage='deep')
print(uber_raw_janjun15.memory_usage())
print('============================================')


# Weather data --------------------------------------------------------------------
# city_attrib = load_file(dir_name_Uber, 'city_attributes.csv')
humidity = load_file(dir_name_Weather, 'humidity.csv')
pressure = load_file(dir_name_Weather, 'pressure.csv')
temperature = load_file(dir_name_Weather, 'temperature.csv')
weather_description = load_file(dir_name_Weather, 'weather_description.csv')
wind_direction = load_file(dir_name_Weather, 'wind_direction.csv')
wind_speed = load_file(dir_name_Weather, 'wind_speed.csv')

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
from functools import reduce
weather_NY = reduce(lambda left, right: pd.merge(left, right , on='datetime',
                   how='outer'), dataframes)

weather_NY.columns = ['datetime', 'humidity', 'pressure', 'temperature', 'weather_description', 'wind_direction', 'wind_speed']

# delete the unnecessary variables
del humidity, pressure, temperature, weather_description, wind_direction, wind_speed

# Weather data (01/01/2015 - 31/06/2015)
weather_NY_15 = weather_NY.loc[(weather_NY['datetime'] < '2015-07')&(weather_NY['datetime'] > '2015')]

del weather_NY

### Print Memory usage ###
print('==Memory Usage: Weather data (old) =========')
weather_NY_15.info(memory_usage='deep')
print(weather_NY_15.memory_usage())
print('============================================')
##########################