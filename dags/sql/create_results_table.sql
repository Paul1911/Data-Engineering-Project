CREATE TABLE IF NOT EXISTS results (
    experiment_id SERIAL PRIMARY KEY,
    experiment_date VARCHAR NOT NULL,
    method VARCHAR NOT NULL,
    best_estimator VARCHAR NOT NULL,
    best_parameters VARCHAR NOT NULL,
    f1_score NUMERIC NOT NULL
);