# coding: utf-8

""" 
Code: TitanicSurvivalPrediction.py
Author: Shota Shirai
Input: train.csv, test.csv
Output: survival_prediction_(model name).csv

Source data: Kaggle - Titanic Survival Prediction (https://www.kaggle.com/c/titanic/data)
Required external codes/modules: provided by 'my_pipeline'

This code processes the titanic passenger data provided by Kaggle based on the analyses 
on the Jupyter Notebook (TitanicSurvivalPrediction_EDA_Model.ipynb) 
in 'EDA_jupyter_notebook' directory. 

Note: Tested Python version 3.8.10
"""
###################################################################################
# Import libraries 
###################################################################################
import os, sys
sys.path.append(os.pardir) # to import files in the parent directory
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier as XGB

# My data pipeline
from my_pipeline.load import load_file
from my_pipeline.dataprofile import data_profile
from my_pipeline.datacleaning import *
from my_pipeline.model import  cls_model_deployer
from my_pipeline.save import save_prediction_csv, save_score_csv

###################################################################################
# Import data 
###################################################################################

# Data is sotred in the "data" directory
df_train = load_file('data/titanic', 'train.csv')
df_test = load_file('data/titanic', 'test.csv')

# Create df_full by concatenating df_train and df_test
df_test["Survived"] = np.nan
df_full = pd.concat([df_train, df_test], axis=0)
df_full.name = 'df_full'

###################################################################################
# Data Profile
###################################################################################
data_profile(df_train)
data_profile(df_test)
# data_profile(df_full)

###################################################################################
# Data Cleaning 
###################################################################################

# === Data Imputation =============================================================

# Cabin, Embarked -----------------------------------------------------------------
col_names = ['Cabin', 'Embarked']
new_str = ['Unknown', 'S']
df_full = repalce_nan_str(df_full, col_names, new_str)
df_full['Cabin_label']=df_full['Cabin'].str.get(0)

# data_profile(df_full)

# Age -----------------------------------------------------------------------------
# Select variables for estimation
df_age_model = df_full[['Age', 'Pclass', 'Sex', 'Parch', 'SibSp']]
df_full = replace_nan_rfr(df_full, df_age_model, 'Age')

# Fare ----------------------------------------------------------------------------
# Fare is correlated with Pclass and replace the missing values with median value of Fare in the same Pclass
fare=df_full.loc[(df_full['Embarked'] == 'S'), 'Fare'].median()
df_full['Fare']=df_full['Fare'].fillna(fare)

###################################################################################
# Data Engineering 
###################################################################################

# Family Size ---------------------------------------------------------------------
df_full['FamilySize']  = df_full['SibSp'] + df_full['Parch'] + 1

# FamilySurvived: Categorize FamilySize according to Survival rate ----------------
# (high(FamilySize = 2-4): 2, low(FamilySize = 1, 5-7): 1, Zero (FamilySize>8): 0)
df_full.loc[(df_full['FamilySize']>=2) & (df_full['FamilySize']<=4), 'FamilySurvived'] = 2
df_full.loc[(df_full['FamilySize']>=5) & (df_full['FamilySize']<=7) | (df_full['FamilySize']==1), 'FamilySurvived'] = 1
df_full.loc[(df_full['FamilySize']>=8), 'FamilySurvived'] = 0

# Name title ----------------------------------------------------------------------
# Extract title from names
df_full['Title'] = df_full['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
pd.unique(df_full['Title'])
df_full['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
df_full['Title'].replace(['Don', 'Sir',  'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)
df_full['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)
df_full['Title'].replace(['Mlle'], 'Miss', inplace=True)
df_full['Title'].replace(['Jonkheer'], 'Master', inplace=True)

# Surname -------------------------------------------------------------------------
df_full['Surname'] = df_full['Name'].map(lambda name:name.split(', ')[0].strip())
df_full['FamilyGroup'] = df_full['Surname'].map(df_full['Surname'].value_counts())

# ---------------------------------------------------------------------------------
# Survival rate of MALE and over 16 years old in the family
Male_Adult = df_full.loc[(df_full['FamilyGroup'] >= 2) & (df_full['Age'] > 16) & (df_full['Sex'] == 'male')] # Sex:0 = male
Male_Adult_list = Male_Adult.groupby('Surname')['Survived'].mean()

# Survival rate of FEMALE and under 16 years old in the family
Female_Child = df_full.loc[(df_full['FamilyGroup'] >= 2) & (df_full['Age'] <= 16) | (df_full['Sex'] == 'female')] # Sex:1 = female
Female_Child_list = Female_Child.groupby('Surname')['Survived'].mean()

# Create Dead list and Survived list
Dead_list=set(Female_Child_list[Female_Child_list.apply(lambda x:x==0)].index)
Survived_list=set(Male_Adult_list[Male_Adult_list.apply(lambda x:x==1)].index)

# print('Dead_list = ', Dead_list)
# print('Survived_list = ', Survived_list)

# Based on dead/survived list, replace Sex, Age, Title to typical data for each case 
df_full.loc[(df_full['Survived'].isnull()) & (df_full['Surname'].apply(lambda x:x in Dead_list)),\
             ['Sex','Age','Title']] = ['male',28.0,'Mr']
df_full.loc[(df_full['Survived'].isnull()) & (df_full['Surname'].apply(lambda x:x in Survived_list)),
             ['Sex','Age','Title']] = ['female',5.0,'Mrs']

# Ticket --------------------------------------------------------------------------
# Ticket group calculates the number of passengers who have the same ticket number
Ticket_Count = dict(df_full['Ticket'].value_counts())
df_full['TicketGroup'] = df_full['Ticket'].map(Ticket_Count)

# Creates groups according to Survival Rate
df_full.loc[(df_full['TicketGroup']>=2) & (df_full['TicketGroup']<=4), 'Ticket_label'] = 2
df_full.loc[(df_full['TicketGroup']>=5) & (df_full['TicketGroup']<=8) | (df_full['TicketGroup']==1), 'Ticket_label'] = 1  
df_full.loc[(df_full['TicketGroup']>=11), 'Ticket_label'] = 0


###################################################################################
# Preprocessing
###################################################################################
# Select features used in estimation
df_full = df_full[['Survived', 'Embarked', 'Age', 'Fare','Pclass', 'Sex', 'Cabin_label', 'FamilySurvived', 'Title', 'Ticket_label']]
# One-hot encoding
df_full =pd.get_dummies(df_full)

# Split dataframe into train and test dataset
train = df_full[df_full['Survived'].notnull()]
X_test  = df_full[df_full['Survived'].isnull()].drop('Survived', axis=1) 

# Dataframe for training and prediction
y_train = train['Survived'] # Survived label (0 or 1)
X_train = train.drop('Survived', axis=1) # Feature variable

###################################################################################
# Model
###################################################################################
# Random seed
rand_seed = 42

# Build models
models = {
    'Logistic Regression': LogisticRegression(n_jobs=-1, max_iter=500)
    , 'Extra Tree':  ETC(
        random_state=rand_seed
        , n_estimators=26
        , max_depth=8
        , min_samples_leaf=2
        , n_jobs=-1
        )
    ,'Random Forest': RFC(
        random_state=rand_seed
        , n_estimators=25
        , max_depth=5
        , max_features='sqrt'
        , n_jobs=-1
        )
    , 'Gradient Boosting': GBC(
        random_state=rand_seed
        , max_depth=5
        , min_samples_leaf=2
        )
    ,'XGBoost': XGB(
        seed=rand_seed
        , eval_metric = 'logloss'
        , use_label_encoder=False
        )
    , 'Linear SVC': LinearSVC(
        random_state=rand_seed
        ,max_iter=10000
    )
    # , 'SVC': SVC(
    #     random_state=rand_seed
    # )
} 


# Grid Search parameters ---------------------------------------------------------
gs_params = {}

# Logistic Regression
gs_params['Logistic Regression'] = None

# Extra Tree
gs_params['Extra Tree'] = {
    'max_depth': [5, 10, 20]
    , 'min_samples_leaf': [2, 5, 10 ,20] 
    }

# Random Forest
gs_params['Random Forest'] = {
    'n_estimators': [1, 5, 10, 20, 25, 50, 100]
    , 'max_depth': [1, 5, 10, 20, 50]
    }

# Gradient Boosting
gs_params['Gradient Boosting'] = {
    'max_depth': [5, 10, 20]
    , 'min_samples_leaf': [2, 5, 10 ,20] 
    }

# XGBoost
gs_params['XGBoost'] = None

# Linear SVC
gs_params['Linear SVC'] = {
    'C': [0.001, 0.01, 0.1, 1, 10]
    }

# SVC
gs_params['SVC'] ={
    'C': [1, 10, 100, 1000, 10000]
    , "gamma": [1, 10, 100]
    # , 'kernel':['poly','rbf']
    }


results = cls_model_deployer(models, X_train, y_train, X_test, gs_params=gs_params)

###################################################################################
# Deploy
###################################################################################
df_id = df_test[['PassengerId']]
file_name = 'TitanicSurvival'
save_prediction_csv(df_id, results, 'Survived', save_filename_head=file_name, res_dir='results_Titanic')

save_score_csv(results, res_dir='results_Titanic')
