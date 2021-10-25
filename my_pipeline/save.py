#coding: utf-8

import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.pardir) # to import files in the parent directory


def save_prediction_csv(df_id, results, res_label, save_filename_head='prediction', res_dir='results'):
    """ Save prediction results 

    """
    for key in results.keys():
        prediction = results[key]['prediction']

        df_out = df_id.join(pd.DataFrame({res_label: prediction.astype(np.int32)}))
        
        df_out.to_csv(res_dir + '/' + save_filename_head + '-' + key + '.csv', index=False)
        print('---------------------------------------------------------')
        print('Prediction result (model:' + key + ') was saved!')


def save_score_csv(results, save_filename_head='ScoreSummary', res_dir='results', show_score=True):
    """ Save score summary for multiple models
    
    Parameters
    ----------
    results (dictionary or list of dictionary): dictionary containing accuracy score for each model
    
    Returns
    -------
    score_summary.csv

    """
    if len(results) > 1:
        cols = ['Models', 'Score']
        df_score = pd.DataFrame(index=[], columns=cols)
        for idx in range(len(results)):
            res_idx = results[idx]
            df_score_temp = pd.DataFrame.from_dict({(i): res_idx[i]['score'] 
                                            for i in res_idx.keys()}
                                            , orient='index').reset_index()
            df_score_temp.columns = ['Models', 'Score']
            df_score = pd.concat([df_score, df_score_temp])
        df_score.reset_index()
    else:
        df_score = pd.DataFrame.from_dict({(i): results[i]['score'] 
                                            for i in results.keys()}
                                            , orient='index').reset_index()
        df_score.columns = ['Models', 'Score']
    df_score.to_csv(res_dir + '/' + save_filename_head + '.csv', index=False)
    if show_score:
        print('============ Accuracy scores ============')
        print(df_score)
        print('================================== SAVED!')
    else: print('============ Accuracy scores were saved!!')