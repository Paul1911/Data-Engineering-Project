CREATE TABLE IF NOT EXISTS target_prediction (
    experiment_id SERIAL PRIMARY KEY,
    experiment_date VARCHAR NOT NULL,
    actual_class VARCHAR NOT NULL,
    predicted_class NUMERIC NOT NULL
);