COPY raw_data
FROM '/maindata/df_raw.csv'
DELIMITER ',' CSV HEADER;
