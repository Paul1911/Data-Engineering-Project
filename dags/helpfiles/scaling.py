import pandas as pd
from sklearn.preprocessing import RobustScaler
from helpfiles.temp_save_load import save_files, load_files

def scale_data():
    '''scales the variables time and amount and saves dataframe'''
    df = load_files(['df'])[0]
    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

    df.drop(['Time','Amount'], axis=1, inplace=True)

    df.name = 'df'

    save_files([df])


