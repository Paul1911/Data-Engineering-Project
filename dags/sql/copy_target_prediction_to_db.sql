COPY target_prediction (experiment_date, actual_class, predicted_class)
FROM '/maindata/target_prediction.csv'
DELIMITER ',' CSV HEADER;
