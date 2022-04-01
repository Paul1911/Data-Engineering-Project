params = {
    "db_engine": "postgresql+psycopg2://airflow:airflow@postgres/airflow",
    "db_schema": "public",
    "db_raw_table": "raw_data",
    "db_results_table": "results",
    "db_target_prediction_table": "target_prediction",
    "test_split_ratio": 0.2,
    "cv_folds": 5,
    "logreg_maxiter": 1000,
    "reduced_dataset_training": True, #If this parameter is set to True, a size-reduced dataset is used for training, which speeds up the process
    "reduced_dataset_fitting": True #If this parameter is set to True, fit_best_model.py uses a smaller dataset for final fitting. If the original dataset is used, it takes several hours. 
}   