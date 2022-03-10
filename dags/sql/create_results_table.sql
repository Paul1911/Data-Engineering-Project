CREATE TABLE IF NOT EXISTS results (
    experiment_id SERIAL PRIMARY KEY,
    experiment_date VARCHAR NOT NULL,
    best_estimator VARCHAR NOT NULL,
    recall_score NUMERIC NOT NULL
);