import math
import pandas as pd
import os, sys
sys.path.append(os.pardir) # to import files in the parent directory
import glob

# @profile
def data_profile_old(df):
    """ Data profile
    
    Parameters
    ----------
    df: dataframe

    Returns
    -------
    df_profile: dataframe for profile info
    """

    df_dtype = pd.DataFrame(df.dtypes)
    # Null count
    df_null = pd.DataFrame(df.isnull().sum())

    # Null ratio (%)
    null_ratio = df.isnull().sum() / df.shape[0] * 100 #Percentage
    null_ratio = null_ratio.apply(lambda x: math.floor(x * 10**2) / 10**2) #rounding
    df_null_ratio = pd.DataFrame(null_ratio)

    # Non-null count
    df_notnull = pd.DataFrame(df.notnull().sum())

    # Unique value/Unique ratio(%)
    df_unique = {}
    df_unique_ratio = {}

    for column in df.columns:
        df_unique[column] = len(df[column].value_counts())
        # df_unique_ratio[column] = len(df[column].value_counts()) / df.shape[0] * 100 #Percentage
        
    df_unique = pd.DataFrame.from_dict(df_unique, orient='index')
    # df_unique_ratio = pd.DataFrame.from_dict(df_unique_ratio, orient='index')
    # df_unique_ratio = df_unique_ratio[0].apply(lambda x: math.floor(x * 10**2) / 10**2) #rounding

    # Create a new dataframe showing the data profile 
    df_profile = pd.concat([df_dtype, df_null, df_null_ratio, df_notnull, df_unique], axis=1).reset_index()
    df_profile.columns = ['Column', 'Data type', 'Null count', 'Null ratio (%)', 'Non-null count', 'Distinct']

    # dataframe name
    df_name = df.name

    print('=== Data profile (' + df_name + ') ========================================')
    print(df_profile)
    print('===========================================================================')

    num_dup = df.duplicated().sum()
    if num_dup > 0:
        print('*************************')
        print('*** WARNING:' + str(num_dup) + 'rows are duplicated')
        print('*************************')
    else: 
        print('*************************')
        print('*** No duplicated row ')
        print('*************************')


    return df_profile
# @profile
def data_profile_new(df):
    
    # Create dataframe to store information
    df_profile = pd.DataFrame(index=df.columns, columns=['Data type', 'Null count', 'Null ratio (%)', 'Non-null count', 'Distinct'])
    
    # Data type
    df_profile['Data type'] = df.dtypes
    
    # Null count
    df_profile['Null count'] = df.isna().sum()
    
    # Null count ratio
    df_profile['Null ratio (%)'] = df_profile['Null count'] / len(df.index) *100
    df_profile['Null ratio (%)'] = df_profile['Null ratio (%)'].round(2)
    
    # Non-null count
    df_profile['Non-null count'] = df.notnull().sum()
    
    # Count of unique values
    df_profile['Distinct'] = df.nunique()

    # Number of duplicated rows
    number_of_duplicates = df.duplicated(keep='first').sum() 
    
    # dataframe name
    try:
        df.name
    except AttributeError: df_name = 'Input Dataframe'
    else: df_name = df.name

    print(''.join(['=== Data profile (', df_name, ') ========================================']))
    print(df_profile)
    print('===========================================================================')

    # number_of_duplicates = len(df)-len(df.drop_duplicates())
    if number_of_duplicates > 0:
        print(' '.join(['*** Warnings:', str(number_of_duplicates), 'rows are duplicated ***']))
    else: print('*** No duplicated row ***')

    return df_profile



if __name__ == '__main__':
    # Data is stored in the "data" directory
    dir_name_Uber = 'data/uber-tlc-foil-response/uber-trip-data'
    dir_name_Weather = 'data/weather'


    # Data is stored in the "data" directory
    dir_name_Uber = 'data/uber-tlc-foil-response/uber-trip-data'
    dir_name_Weather = 'data/weather'

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
    # Concatenate the dataframes in the list
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

    # df_uber.name = 'Uber rides (2015)'
    df_weather.name = 'Weather data in NY (2015)'
    data_profile_new(df_uber)