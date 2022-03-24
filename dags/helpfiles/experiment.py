import numpy as np 
import pandas as pd 

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

# Other Libraries
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

from helpfiles.temp_save_load import save_files, load_files
import helpfiles.ml_pipeline_config as configurations

def experiment():
    cv_folds = configurations.params['cv_folds']

    new_df = load_files(['new_df'])[0]

    X = new_df.drop('Class', axis=1)
    y = new_df['Class']
    
    X_val = load_files(['X_val'])[0]
    y_val = load_files(['y_val'])[0]

    #Predict Y
    """     y_pred_log_reg = log_reg.predict(X_val)
    y_pred_SVM = svc.predict(X_val)
    f1_log_reg = f1_score(y_val, y_pred_log_reg)
    f1_SVM = f1_score(y_val, y_pred_SVM) """

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
    gs = GridSearchCV(pipeline, params, cv=cv_folds, n_jobs=-1, scoring='recall').fit(X,y)
    print(gs.best_params_)

    # Predict Y
    y_pred= gs.best_estimator_.predict(X_val)

    # Calculate recall
    recall_val = recall_score(y_val, y_pred)
    print("Recall validation result:" + str(recall_val))

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    #Creation of results table 
    results = pd.DataFrame([[
        now,
        gs.best_params_.get('classifier'),
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