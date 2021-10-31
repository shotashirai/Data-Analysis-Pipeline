# coding: utf-8

# Visualisation
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.io as pio
# import seaborn as sns
plt.rcParams['font.size'] = '16'# Set general font size


def plot_feat_imp(results, target_list, num_feat=10, label_str=''):
    df_res = results.copy()
    for target in target_list:
        print(target)
        #Extract data
        feat_imp_target = results[target]['feature_importances']
        X_train_target = results[target]['X_train']
        
        df_feat_imp = pd.DataFrame({'col': X_train_target.columns,'imp':feat_imp_target})
        df_feat_imp = df_feat_imp.sort_values(by='imp',ascending=False)
        if df_feat_imp.shape[0] > num_feat:
            df_feat_imp = df_feat_imp.iloc[:10]
            
        # Plot
        fig, ax = plt.subplots(figsize=(20,5))
        
        bar = ax.barh(df_feat_imp.col, df_feat_imp.imp, 0.6
                     , color=mcolors.TABLEAU_COLORS)
        ax.set_ylabel('Feature', fontsize=16)
        ax.set_xlabel('Feature Importances')
        ax.set_title('Feature Importances - '+target+label_str)
        ax.bar_label(bar, fmt='%.02f')
        
        ax.invert_yaxis()
        
        figname='Feature Importances - '+target+label_str
        plt.savefig(('images/'+figname+'.png'),  bbox_inches='tight')
        fig.tight_layout() 
        plt.show()

def plot_feat_imp_multi(results, target_list, num_feat=10, label_str=''):
    df_res = results.copy()
    fig, axes = plt.subplots(len(target_list), 1, figsize=(20,8*len(target_list)))
    for target, ax in zip(target_list, axes):
        print(target)
        #Extract data
        feat_imp_target = results[target]['feature_importances']
        X_train_target = results[target]['X_train']
        
        df_feat_imp = pd.DataFrame({'col': X_train_target.columns,'imp':feat_imp_target})
        df_feat_imp = df_feat_imp.sort_values(by='imp',ascending=False)
        if df_feat_imp.shape[0] > num_feat:
            df_feat_imp = df_feat_imp.iloc[:10]
            
        # Plot
        
        
        bar = ax.barh(df_feat_imp.col, df_feat_imp.imp, 0.6
                     , color=mcolors.TABLEAU_COLORS)
        ax.set_ylabel('Feature', fontsize=16)
        ax.set_xlabel('Feature Importances')
        ax.set_title('Feature Importances - '+target+label_str)
        ax.bar_label(bar, fmt='%.02f')
        
        ax.invert_yaxis()
        
    figname='Feature Importances - multi'+label_str
    plt.savefig(('images/'+figname+'.png'),  bbox_inches='tight')
    plt.show()