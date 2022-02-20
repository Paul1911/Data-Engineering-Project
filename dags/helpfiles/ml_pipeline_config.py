params = {
    "db_engine": "postgresql+psycopg2://airflow:airflow@postgres/airflow",
    "db_schema": "public",
    #"db_experiments_table": "experiments",
    "db_raw_table": "raw_data",
    "db_results_table": "results",
    "test_split_ratio": 0.2,
    "cv_folds": 5,
    "logreg_maxiter": 1000 #Todo: check what you need here and adjust file
}   