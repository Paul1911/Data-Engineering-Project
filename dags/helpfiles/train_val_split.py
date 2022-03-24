import pandas as pd
from sklearn.model_selection import train_test_split
from helpfiles.temp_save_load import save_files, load_files
import helpfiles.ml_pipeline_config as configurations

split_ratio = configurations.params['test_split_ratio']

def split_data_train_val():
    '''Splits dataset according to provided split ratio in config file'''
    df = load_files(['df'])[0]

    # Splitting
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_ratio) #todo: add configurations here

    # Dataframe is concatenated again as the full dataset is needed and split again during cross validation. _val dataframes are needed for the final evaluation. 
    df = pd.concat([X_train, y_train],axis = 1)

    # Naming
    df.name = 'df'
    X_val.name = 'X_val'
    y_val.name = 'y_val'

    # Saving
    save_files([df, X_val, y_val])



