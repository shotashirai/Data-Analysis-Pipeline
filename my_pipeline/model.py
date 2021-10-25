# coding: utf-8
import numpy as np
from sklearn.linear_model import LinearRegression
from boruta import BorutaPy
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

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




        