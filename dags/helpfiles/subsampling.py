import pandas as pd
from helpfiles.temp_save_load import save_files, load_files

def undersample_data():
    '''This function reads the df from temp, shuffles it, filters fraud cases and adds similar number 
    of non fraud cases. Final df is then saved again as new_df'''

    df = load_files(['df'])[0]
    df2 = df.sample(frac=1) # this shuffles the initial df

    # Filter fraud cases and concat equal amount of non-fraud-cases
    fraud_df = df2.loc[df['Class'] == 1]
    non_fraud_df = df2.loc[df['Class'] == 0][:len(fraud_df)]

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

    # Shuffle dataframe rows
    new_df = normal_distributed_df.sample(frac=1, random_state=42)
    new_df.name = 'new_df'

    # Save
    save_files([new_df])