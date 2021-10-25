# coding: utf-8
from datetime import datetime
import pandas as pd
import sys

def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

def memory_usage(var, lower_limit=0):
    ''' Memory Usage
    This code provides information about the memory usage
    
    Parameters
    ----------
    var = dir()
    lower_limit (default:0, optional): define the minimam value of the memory usage displayed
    
    Return
    ------
    print memory usage
    '''
    
    # input: var = dir()
    print("{}{: >25}{}{: >10}{}".format('|','Variable Name','|','Memory','|'))
    print(" ------------------------------------ ")
    for var_name in var:
        if not var_name.startswith("_") and sys.getsizeof(eval(var_name)) > lower_limit:
            print("{}{: >25}{}{: >10}{}".format('|',var_name,'|',sys.getsizeof(eval(var_name)),'|'))
    return

def datatime_format(df, col_datetime, get_new_cols=True):
    df['datetime'] = pd.to_datetime(df[col_datetime], format='%Y-%m-%d %H:%M:%S')
    if get_new_cols:
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['weekday'] = df['datetime'].dt.dayofweek
        df['weekday_num'] = df['datetime'].dt.day
    
    return df

def print_timeperiod(df, datetime_col):
    df_name = df.name
    
    print('Time Period - ' + df_name)
    print('Min date: %s' % df[datetime_col].min())
    print('Max date: %s' % df[datetime_col].max())