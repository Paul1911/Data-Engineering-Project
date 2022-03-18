import numpy as np
import pandas as pd
from helpfiles.temp_save_load import save_files

def load_data():
    '''Loads credit fraud data and doubles its observations until 1,000,000 lines are reached 
    as this is a requirement for the project. Afterwards, it is saved in temp'''
    df = pd.read_csv("/opt/airflow/data/creditcard.csv").head(10000) # Todo: remove facilitation
    #while len(df.index) < 1000000:
    #    df = pd.concat([df,df])
    df.reset_index(drop = True, inplace = True)
    df.name = 'df'

    save_files([df])