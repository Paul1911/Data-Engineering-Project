# Todo: Clean up imports
# Todo: Clean up file

import numpy as np # linear algebra
import pandas as pd # data processing
#import tensorflow as tf
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA, TruncatedSVD
#import matplotlib.patches as mpatches
#import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import collections

# Other Libraries
#from sklearn.model_selection import train_test_split
#from sklearn.pipeline import make_pipeline
#from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import NearMiss
#from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
#from collections import Counter
#from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

from helpfiles.temp_save_load import save_files, load_files
import helpfiles.ml_pipeline_config as configurations

def experiment():
    new_df = load_files(['new_df'])[0]

    X = new_df.drop('Class', axis=1)
    y = new_df['Class']

    #classifiers = {
    #"LogisticRegression": LogisticRegression(),
    #"Support Vector Classifier": SVC()
    #}

    #for key, classifier in classifiers.items():
    #classifier.fit(X, y)
    #training_score = cross_val_score(classifier, X, y, cv=5, scoring = 'f1')
    #print("Classifiers: ", classifier.__class__.__name__, "Has a training F1 of", round(training_score.mean(), 2) * 100)
    
    
    # Logistic Regression 
    log_reg_maxiter = configurations.params["logreg_maxiter"]
    log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]} #todo: put this into pipeline config
    # Grid search
    grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
    #grid_log_reg = GridSearchCV(LogisticRegression(max_iter = log_reg_maxiter), log_reg_params) # todo: see whether you can use this line
    grid_log_reg.fit(X, y)
    # Logistic Regression estimator:
    log_reg = grid_log_reg.best_estimator_

    # Support Vector Classifier
    svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']} #todo: put this into pipeline config
    # Grid search
    grid_svc = GridSearchCV(SVC(), svc_params)
    grid_svc.fit(X, y)
    # SVC best estimator
    svc = grid_svc.best_estimator_

    # Evaluation
    # Logistic Regression 
    log_reg_score = cross_val_score(log_reg, X, y, cv=5, scoring = 'f1')
    print('Logistic Regression Cross Validation F1 score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')
    print("Best Parameters Logistic Regression: " + str(grid_log_reg.best_params_))
    # SVC
    svc_score = cross_val_score(svc, X, y, cv=5, scoring = 'f1')
    print('Support Vector Classifier Cross Validation F1 score', round(svc_score.mean() * 100, 2).astype(str) + '%')
    print("Best Parameters SVM: " + str(grid_svc.best_params_))
    
    X_val = load_files(['X_val'])[0]
    y_val = load_files(['y_val'])[0]

    #Predict Y
    y_pred_log_reg = log_reg.predict(X_val)
    y_pred_SVM = svc.predict(X_val)
    f1_log_reg = f1_score(y_val, y_pred_log_reg)
    f1_SVM = f1_score(y_val, y_pred_SVM)

    #******************************************************************************************************

    # Initialize the estimators
    log_reg_maxiter = configurations.params["logreg_maxiter"]

    clf1 = SVC()
    clf2 = LogisticRegression(max_iter = log_reg_maxiter)
    clf3 = KNeighborsClassifier()
    clf4 = GradientBoostingClassifier()

    # Initialize hyperparameters for each dictionary

    param1 = {}
    param1['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
    param1['classifier__kernel'] = ['rbf', 'poly', 'sigmoid', 'linear']
    #param1['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
    param1['classifier'] = [clf1]

    param2 = {}
    param2['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
    param2['classifier__penalty'] = ['l1', 'l2']
    #param2['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
    param2['classifier'] = [clf2]

    param3 = {}
    param3['classifier__n_neighbors'] = [2,5,10,25,50]
    param3['classifier'] = [clf3]

    param4 = {}
    param4['classifier__n_estimators'] = [10, 50, 100, 250]
    param4['classifier__max_depth'] = [5, 10, 20]
    param4['classifier'] = [clf4]



    pipeline = Pipeline([('classifier', clf1)])
    params = [param1, param2, param3, param4]

    # Train the grid search model
    gs = GridSearchCV(pipeline, params, cv=5, n_jobs=-1, scoring='recall').fit(X,y)
    print(gs.best_params_)

    # Predict Y
    y_pred= gs.best_estimator_.predict(X_val)

    # Calculate recall
    recall_val = recall_score(y_val, y_pred)
    print("Recall validation result:" + str(recall_val))

    #******************************************************************************************************

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_reg_results = pd.DataFrame([[
        now,
        'Logistic Regression',
        grid_log_reg.best_estimator_,
        grid_log_reg.best_params_,
        f1_log_reg
        ]],
        columns = [
            'experiment_date',
            'method',
            'best_estimator',
            'best_parameters',
            'f1_score'])
    log_reg_results

    svc_results = pd.DataFrame([[
        now,
        'Support Vector Classifier',
        grid_svc.best_estimator_,
        grid_svc.best_params_,
        f1_SVM
        ]],
        columns = [
            'experiment_date',
            'method',
            'best_estimator',
            'best_parameters',
            'f1_score'])

    results2 = pd.concat([log_reg_results,svc_results])
    results2.name = 'results2'

    save_files([results2])

    #******************************************************************************************************
    #Creation of results table 
    results = pd.DataFrame([[
        now,
        gs.best_params_['classifier'],
        recall_val
        ]],
        columns = [
            'experiment_date',
            'best_estimator',
            'recall_score'])
    
    results.name = 'results'
    save_files([results])

    #Creation of the target prediction table for metrics calculation
    y_pred_series = pd.Series(y_pred)

    target_prediction = pd.concat([y_val.reset_index(),y_pred_series], axis = 1, ignore_index=True)
    target_prediction[0] = now
    target_prediction.columns = [
        'experiment_date',
        'actual_class',
        'predicted_class']

    target_prediction.name = 'target_prediction'
    save_files([target_prediction])
    
