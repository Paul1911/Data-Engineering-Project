from airflow.models import DAG

from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.task_group import TaskGroup

from datetime import datetime

from helpfiles.experiment import experiment
from helpfiles.import_data import load_data
from helpfiles.postgres_save import prepare_raw_data, save_results_data
from helpfiles.scaling import scale_data
from helpfiles.subsampling import undersample_data
from helpfiles.train_val_split import split_data_train_val
# from helpfiles.validity_checks import something

default_args= {
    'owner': 'Paul Friedrich',
    'email_on_failure': False,
    'email': ['paul.friedrich@iubh.de'],
    'start_date': datetime(2022, 2, 13)
}

with DAG(
    "ml_pipeline",
    description='Credit Fraud Classification',
    schedule_interval='@daily',
    default_args=default_args, 
    catchup=False) as dag:

    # task: 1
    #with TaskGroup('creating_storage_structures') as creating_storage_structures:

    # Parallel creation statements for tables can throw errors in PostgreSQL, therefore it has to be done sequentially for the time being
    # task: 1.1
    creating_experiment_tracking_table = PostgresOperator(
        task_id="creating_experiment_tracking_table",
        postgres_conn_id='postgres_default',
        sql='sql/create_results_table.sql'
    )

    # task: 1.2
    creating_batch_data_table = PostgresOperator(
        task_id="creating_batch_data_table",
        postgres_conn_id='postgres_default',
        sql='sql/create_raw_data_table.sql'
    )
    # task: 1.3
    creating_target_prediction_table = PostgresOperator(
        task_id="creating_target_prediction_table",
        postgres_conn_id='postgres_default',
        sql='sql/create_target_prediction_table.sql'
    )
    
    
    # task: 2
    fetching_data = PythonOperator(
        task_id='fetching_data',
        python_callable= load_data
    )

    # task: 3.1
    prepare_raw_data = PythonOperator(
        task_id = 'prepare_raw_data',
        python_callable = prepare_raw_data
    )

    # task: 3.2 which is task 3 new with copy
    saving_raw_data = PostgresOperator(
        task_id="saving_raw_data",
        postgres_conn_id='postgres_default',
        sql='sql/copy_df_to_db.sql'
    )

    # task: 4
    scaling = PythonOperator(
        task_id = 'scaling',
        python_callable = scale_data
    )

    # task: 5
    splitting = PythonOperator(
        task_id = 'splitting',
        python_callable = split_data_train_val
    )

    # task: 6
    undersampling = PythonOperator(
        task_id = 'undersampling',
        python_callable = undersample_data
    )

    # task: 7
    experimenting = PythonOperator(
        task_id = 'experimenting',
        python_callable = experiment
    )

    # task: 8
    with TaskGroup('experiment_csv_to_db') as experiment_csv_to_db:

        # task: 8.1
        saving_result_data = PythonOperator(
            task_id = 'saving_result_data',
            python_callable = save_results_data
        )

        # task: 8.2
        saving_target_prediction_data= PostgresOperator(
        task_id="saving_target_prediction_data",
        postgres_conn_id='postgres_default',
        sql='sql/copy_target_prediction_to_db.sql'
    )

    # Old workflow, can be reinstated when parallel processing of SQL CREATE TABLE statements is available 
    #creating_storage_structures >> fetching_data >> prepare_raw_data >> saving_raw_data >> scaling >> splitting >> undersampling >> experimenting >> experiment_csv_to_db
    # New workflow
    creating_experiment_tracking_table >> creating_batch_data_table >> creating_target_prediction_table >> fetching_data >> prepare_raw_data >> saving_raw_data >> scaling >> splitting >> undersampling >> experimenting >> experiment_csv_to_db

    #creating_experiment_tracking_table >> creating_batch_data_table >> creating_target_prediction_table >> fetching_data >> prepare_raw_data >> scaling >> splitting >> undersampling >> experimenting >> experiment_csv_to_db