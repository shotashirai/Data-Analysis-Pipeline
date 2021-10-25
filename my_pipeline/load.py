# coding: utf-8

import pandas as pd

def load_file(dir_name, file_name):
    """ Load Files

    Parameters
    ----------
    dir_name (str): Directory name containing data file (csv)
    file_name (str): File name (.csv)

    Returns
    -------
    df: pandas dataframe with a name (df.name = 'abc')

    """
    
    df = pd.read_csv(dir_name + '/' + file_name)
    df.name = file_name
    return df