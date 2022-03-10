COPY target_prediction
FROM '/maindata/target_prediction.csv'
DELIMITER ',' CSV HEADER;
