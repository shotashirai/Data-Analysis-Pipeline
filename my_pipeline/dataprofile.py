import pandas as pd

def data_profile(df):
    
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

