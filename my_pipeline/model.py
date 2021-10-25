# coding: utf-8
import numpy as np
from sklearn.linear_model import LinearRegression
from boruta import BorutaPy
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)

class BinaryLinearRegression:
    """ Linear Regression model for binary classification
: modified linear regression model. To use for binary classification (1 or 0), a threshold is used.
    """
    
    def __init__(self):
        self.linR = LinearRegression(n_jobs=-1)

    def fit(self, X_fit, Y_fit):
        return self.linR.fit(X_fit, Y_fit)

    def predict(self,X_test, threshold=0.5):
        prediction = self.linR.predict(X_test)
        binary_prediction = np.where(prediction > threshold, 1, 0)
        return binary_prediction


def cls_model_deployer(models, X_train, y_train, X_test
                , feature_select=True, gs_params=None, cv=10):
    
    """ Model Deployer:
    train models and return prediction results 

    Parameters
    ----------
    models (dictionary): a list of models used for prediction in dictionary 
    X_train (dataframe): training data
    y_train (dataframe): training label
    X_test (dataframe): test data to be predicted
    feature_select (Boolean, Default: True) : Feature selection by BorutaPy
    gs_params(Boolean, Default: True) : GridSearch Flag

    Returns
    -------
    results: prediction results, feature_importance, feature_name, best_estimator, score
    """

    results = {}

    if gs_params is None:
        gs_params = {k:None for k in models.keys()}

    for key in models.keys():
        ##################################################################################
        # Feature Selection 
        ##################################################################################
        if feature_select:
            print('===== Feature selection (' + key + ') by Boruta starts... =====')
            feature_selector = BorutaPy(
                models[key]
                , n_estimators='auto'
                , alpha=0.1
                , max_iter=100
                , random_state=42
                , verbose=0                
            )
            feature_selector.fit(X_train.values, y_train.values)
            print('Selected Feature:') 
            print(X_train.columns[feature_selector.support_])
            print('===== Feature selection was done... =====')
            # Select only selected feature
            X_train_selected = X_train.iloc[:, feature_selector.support_]
            X_test_selected = X_test.iloc[:, feature_selector.support_]
        else:
            X_train_selected = X_train
            X_test_selected = X_test

        ##################################################################################
        # GridSearchCV 
        ##################################################################################
        if gs_params[key] is not None:
            
            gs_params_key = gs_params[key].copy()
            # GridSearch CV
            gscv = GridSearchCV(estimator=models[key]
            , param_grid=gs_params_key
            , cv=cv
            , refit=True
            , n_jobs=-1
            )

            #train model
            gscv.fit(X_train_selected, y_train)

            # Show the results
            print(str(key) + '(with GridSearchCV)')
            print('Best Score: {}' .format(np.round(gscv.best_score_, 4))) # Best score
            print('Best Parameters: ', gscv.best_params_) # Best paramters in Grid Search
            print('======================================================')

            # Results
            if hasattr(gscv.best_estimator_, 'feature_importances_'):
                results[key] = {'score': gscv.best_score_
                                , 'feature_name': X_train_selected.columns
                                , 'estimator':gscv.best_estimator_
                                , 'feature_importances':gscv.best_estimator_.feature_importances_
                                , 'prediction':gscv.predict(X_test_selected)
                                }
            else:
                results[key] = {'score': gscv.best_score_
                                , 'feature_name':  X_train_selected.columns
                                , 'estimator':gscv.best_estimator_
                                , 'feature_importances':None
                                , 'prediction':gscv.predict(X_test_selected)
                                }
        ##################################################################################
        else:
            # Model
            clf = models[key]
            
            # train model
            clf.fit(X_train_selected, y_train)
            
            # Cross validation
            cv_result = cross_validate(clf, X_train_selected, y_train, cv=cv)
            
            # Show the results
            print(str(key))
            print("Mean Score:{}".format(np.round(np.mean(cv_result['test_score']), 4))) # show the mean score
            print("Mean std:{}".format(np.round(np.std(cv_result['test_score']), 4))) # Standard deviation of the score
            print('======================================================')

            if hasattr(clf, 'feature_importances_'):
                results[key] = {'score': np.mean(cv_result['test_score'])
                                , 'feature_name': X_train_selected.columns
                                , 'estimator': models[key]
                                , 'feature_importances':clf.feature_importances_
                                , 'prediction':clf.predict(X_test_selected)
                                }
            else:
                results[key] = {'score': np.mean(cv_result['test_score'])
                                , 'feature_name': X_train_selected.columns
                                , 'estimator':models[key]
                                , 'feature_importances':None
                                , 'prediction':clf.predict(X_test)
                                }
    return results

# Custom function for time series analys using XGBoost
def prediction_XGBoost(df_train, df_test, model, target_list
                       , feature_select=False, log_transform=False, include_borough=False):
    
    def _mape(true, pred): 
        true, pred = np.array(true), np.array(pred)
        return np.mean(np.abs((true - pred) / true)) * 100
    
    results = {}
    for target in target_list:
#         print(model)
        print('Forecasting:'+target)
        new_df_train = df_train.copy()
        new_df_test = df_test.copy()
        
        # Define Y_train and Y_test (Target label)
        Y_train = new_df_train[target]
        Y_test = new_df_test[target]
        
        
        drop_list = target_list.copy() # list of dropped variables

        ##----- When rides variables for majour borough are not included ------------------##
        if include_borough:
            del_borough_list = target_list.copy()
            del_borough_list.remove(target)
            for del_borough in del_borough_list:
                all_columns_df_model = list(df_model.columns) # all columns of df_train
                drop_borough_col = df_train.columns.str.contains(del_borough+'_') #boolean
                drop_borough_list = list(compress(col_name_df_model, drop_borough_col))
                # Add dropped columns on the drop_list
                drop_list = drop_list + drop_borough_list
        ##---------------------------------------------------------------------------------##
        
        # Define X_train and X_test (Features)
        X_train = new_df_train.drop(drop_list, axis=1)
        X_test = new_df_test.drop(drop_list, axis=1)
        
        
        
        # Feature selection----------------------------------------------------------------##
        if feature_select:
            feature_selector =BorutaPy(model
                                   ,n_estimators='auto' 
                                   ,verbose=0 # 0: no output,1: displays iteration number,2: which features have been selected already
                                   ,alpha=0.1
                                   ,max_iter=100
                                   ,random_state=42
                                  )
            feature_selector.fit(X_train.values, Y_train.values)
            # Select only selected feature
            X_train = X_train.iloc[:,feature_selector.support_]
            X_test = X_test.iloc[:,feature_selector.support_]
        
        # ---------------------------------------------------------------------------------##
        
        
        # Fitting
        model.fit(
            X_train, Y_train, 
            eval_set=[(X_train, Y_train), (X_test, Y_test)], 
            verbose=0
#             early_stopping_rounds = 10
        )
        
        # Prediction
        Y_predict = model.predict(X_test)
        if log_transform:
            # Accuracy (RMSE)
            mse = mean_squared_error(Y_test,Y_predict)
            rmse = np.sqrt(mse)
            print('RMLSE: ' + str(np.round(rmse, 2)))

            # Accuracy (MAPE)
            mape = _mape(np.expm1(Y_test), np.expm1(Y_predict))
            print('MAPE: ' + str(np.round(mape, 2)) +'%')
        
            # Results
            results[target] = {'feature_importances': model.feature_importances_
                               ,'mape': mape
                               ,'rmse': rmse
                               ,'X_train': X_train
                               ,'X_test': X_test
                               ,'Y_train': np.expm1(Y_train)
                               ,'Y_test': np.expm1(Y_test)
                               ,'Y_predict': np.expm1(Y_predict)
                          }
            
        else:
            
            # Accuracy (RMSE)
            mse = mean_squared_error(Y_test, Y_predict)
            rmse = np.sqrt(mse)
            print('RMSE: ' + str(np.round(rmse, 2)))

            # Accuracy (MAPE)
            mape = _mape(Y_test, Y_predict)
            print('MAPE: ' + str(np.round(mape, 2)) +'%')
            
            # Results
            results[target] = {'feature_importances': model.feature_importances_
                               ,'mape': mape
                               ,'rmse': rmse
                               ,'X_train': X_train
                               ,'X_test': X_test
                               ,'Y_train': Y_train
                               ,'Y_test': Y_test
                               ,'Y_predict': Y_predict
                              }
            
    return results

        