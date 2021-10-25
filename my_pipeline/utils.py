# coding: utf-8
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