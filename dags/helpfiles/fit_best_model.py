import numpy as np 
import pandas as pd 

# Classifier Libraries
# We must import all libraries we might need
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
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
    df_raw, new_df, results = load_files(['df_raw', 'new_df', 'results'])
    creditcard = pd.read_csv("/opt/airflow/data/creditcard.csv")
    df_raw = df_raw.drop(['datetime_write_query'], axis = 1)
    model = eval(results['best_estimator'].values[0])

    scaler = ColumnTransformer([('robustscaler',
     RobustScaler(),
     ['Time','Amount']
     )], remainder='passthrough')

    pipe = Pipeline([('robustscaler', scaler),('model', model)])

    #using the training data for model export
    X = creditcard.drop('Class', axis=1)
    y = creditcard['Class']

    #using the full dataset for model export
    #Attention: This takes very long (1.5-2hrs with 4GB RAM) as we have such a large dataset
    #X = df_raw.drop(['Class'], axis = 1)
    #y = df_raw['Class']

    pipe.fit(X,y)

    #save model
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = 'model_' + now + '.pkl'
    joblib.dump(pipe, '/opt/airflow/models/' + filename, compress=1)
