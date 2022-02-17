import pandas as pd
import os.path

def save_files(df_list):
    '''
    accepts dataframe list as input
    saves each dataframe in the tmp folder as csv
    the file name corresponds to the dataframe "name" attribute
    '''
    #[ df.to_parquet('/opt/airflow/data/' + df.name + '.parquet' , index=False) for df in df_list ]
    [ df.to_csv('/opt/airflow/data/' + df.name + '.csv' , sep=',', index=False) for df in df_list ]


def load_files(names_list):
    '''
    accepts a list of names (str) as input
    load each csv file from the tmp folder with the input names
    returns a list of loaded dataframes
    '''
    df_list = []
    #[ df_list.append(pd.read_parquet("/opt/airflow/data/" + name + ".parquet")) for name in names_list if os.path.isfile('/opt/airflow/data/' + name + '.parquet') ]
    [ df_list.append(pd.read_csv("/opt/airflow/data/" + name + ".csv")) for name in names_list if os.path.isfile('/opt/airflow/data/' + name + '.csv') ]
    return df_list
# Todo: Check whether temp saving as .csv is ok. Bare in mind that to_csv overwrites file names, which you definitely need!