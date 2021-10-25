import math
import pandas as pd

def data_profile(df):
    """ Data profile
    
    Parmerters
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
        df_unique_ratio[column] = len(df[column].value_counts()) / df.shape[0] * 100 #Percentage
        
    df_unique = pd.DataFrame.from_dict(df_unique, orient='index')
    df_unique_ratio = pd.DataFrame.from_dict(df_unique_ratio, orient='index')
    df_unique_ratio = df_unique_ratio[0].apply(lambda x: math.floor(x * 10**2) / 10**2) #rounding

    # Create a new dataframe showing the data profile 
    df_profile = pd.concat([df_dtype, df_null, df_null_ratio, df_notnull, df_unique, df_unique_ratio], axis=1).reset_index()
    df_profile.columns = ['Column', 'Data type', 'Null count', 'Null ratio (%)', 'Non-null count', 'Distinct', 'Distinct (%)']

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

