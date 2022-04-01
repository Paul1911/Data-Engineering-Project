import numpy as np
import pandas as pd
from helpfiles.temp_save_load import save_files
import helpfiles.ml_pipeline_config as configurations

def load_data():
    '''Loads credit fraud data and doubles its observations until 1,000,000 lines are reached 
    as this is a requirement for the project. Afterwards, it is saved in temp'''
    reduced_dataset = configurations.params['reduced_dataset_training']
    if reduced_dataset == False:
        df = pd.read_csv("/opt/airflow/data/creditcard.csv")
        while len(df.index) < 1000000:
            df = pd.concat([df,df])
    elif reduced_dataset == True:
        df = pd.read_csv("/opt/airflow/data/creditcard.csv").head(10000)
    else:
        raise ValueError("reduced_dataset configuration is not True or False")

    df.reset_index(drop = True, inplace = True)
    df.name = 'df'

    save_files([df])