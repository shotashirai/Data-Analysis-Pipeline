# coding: utf-8

import pandas as pd

# def gen_lagdata(df, lag_vars, lags):
#     df_model = df.copy()
#     df_model['time_block_num'] = df_model.index+1

#     for lag in lags:
#         df_lag = df_model.copy()
#         df_lag.time_block_num += lag
#         # subset only the lag variable required
#         df_lag = df_lag[['time_block_num']+lag_vars]
#         df_lag.columns = ['time_block_num']+[lag_feat+'_lag_'+str(lag) for lag_feat in lag_vars]
    
#         df_model = pd.merge(df_model, df_lag, on=['time_block_num'], how='left')

#     df_model = df_model.drop(range(0, max(lags))).reset_index()
#     df_model = df_model.drop(['index'], axis=1)

#     return df_model

def gen_lagdata(df, columns, lags, drop_NaN=True):

    for lag in lags:
        # Append shifted columns
        df[['{}_lag{}'.format(col, lag) for col in columns]] = df[columns].shift(lag)
    
    # Drop rows containing NaN values
    if drop_NaN:
        df.dropna(inplace= True)
    
    return df


if __name__ == '__main__':
    df = pd.DataFrame({
                'a': [10, 20, 15, 30, 45],
                'b': [13, 23, 18, 33, 48],
                'c': [17, 27, 22, 37, 52]},
                index=pd.date_range("2020-01-01", "2020-01-05"))
    
    cols =  ['a', 'b']
    df = gen_lagdata(df=df, columns=cols, lags=[1, 2])
    print(df)


