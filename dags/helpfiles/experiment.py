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
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
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

    results = pd.concat([log_reg_results,svc_results])
    results.name = 'results'

    save_files([results])
    
