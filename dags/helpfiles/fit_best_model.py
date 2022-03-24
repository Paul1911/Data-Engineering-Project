import numpy as np 
import pandas as pd 

# Classifier Libraries
# We must import all libraries we might need
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from datetime import datetime
import joblib

# Other Libraries
import warnings
warnings.filterwarnings("ignore")

from helpfiles.temp_save_load import load_files
import helpfiles.ml_pipeline_config as configurations

def fit_export_best_model():
    df_raw, results = load_files(['df_raw', 'results'])
    df_raw = df_raw.drop(['datetime_write_query'], axis = 1)
    model = eval(results['best_estimator'].values[0])

    scaler = ColumnTransformer([('robustscaler',
     RobustScaler(),
     ['Time','Amount']
     )], remainder='passthrough')

    pipe = Pipeline([('robustscaler', scaler),('model', model)])

    pipe.fit(df_raw.drop(['Class'], axis = 1), df_raw['Class'])

    print(metrics.classification_report(df_raw['Class'], pipe.predict(df_raw)))
    print("Attention: nice to know metrics, but its meaning is limited as we use the same data for training and testing")

    #save model
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = 'model_' + now + '.pkl'
    joblib.dump(pipe, '/opt/airflow/models/' + filename, compress=1)
