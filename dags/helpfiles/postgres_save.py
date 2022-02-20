import pandas as pd
from sqlalchemy import create_engine
from helpfiles.temp_save_load import load_files, save_files
import helpfiles.ml_pipeline_config as configurations
from datetime import datetime 

db_engine = configurations.params["db_engine"]
db_schema = configurations.params["db_schema"]
table_raw = configurations.params["db_raw_table"] 
table_results = configurations.params["db_results_table"]

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_raw_data():
    df = load_files(['df'])[0]
    df_raw = df
    df_raw["datetime_write_query"] = now

    df_raw.name = 'df_raw'

    save_files([df_raw])
    #engine = create_engine(db_engine)
    #df.to_sql(table_raw, engine, schema=db_schema, if_exists='replace', index=False, chunksize = 1000, method = 'multi')
    #df.to_sql(table_raw, engine, schema=db_schema, if_exists='replace', index=False)

def save_results_data():
    results = load_files(['results'])[0]
    #results["datetime_write_query"] = now #todo: add column correctly
    engine = create_engine(db_engine)
    results.to_sql(table_results, engine, schema=db_schema, if_exists='append', index=False)

