import dash
import dash_core_components as dcc
import dash_html_components as html
import flask
from sklearn import metrics 

import pandas as pd
from sqlalchemy import create_engine
import helpfiles.ml_pipeline_config as configurations 

db_engine = configurations.params["db_engine"]
db_schema = configurations.params["db_schema"]
table_raw = configurations.params["db_raw_table"] 
table_results = configurations.params["db_results_table"]
table_target_prediction = configurations.params["db_target_prediction_table"] 

engine = create_engine(db_engine)


app= flask.Flask(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
dash_app = dash.Dash(__name__, server=app, external_stylesheets=external_stylesheets)
dash_app.config.suppress_callback_exceptions = True

#we define this intermediate function to allow for data updates by refreshing. Otherwise, the SQL-Query would not be refreshed. 
def serve_layout():
    #Read tables
    #raw_data_from_db = pd.read_sql_table('raw_data', con = engine, columns = ['Class']) 
    '''This approach of raw data visualization is not working as nginx times out. Even when working with SQL queries, it takes ages.
    For the time being, we stay with the metrics visualization.'''
    target_prediction_from_db = pd.read_sql_table('target_prediction', con = engine)

    #Raw data check
    #exp_date = pd.read_sql('SELECT DISTINCT Datetime_write FROM raw_data', engine)
    #print(exp_date)
    #observation_count = []
    #nan_rows = []
    #fraud_count = []
    #for timestamp in raw_data_from_db['Datetime_write'].unique():
    #    df = raw_data_from_db[raw_data_from_db['Datetime_write'] == timestamp]
    #    observation_count.append(len(df.index))
    #    nan_rows.append(len(df[df.isna().any(axis=1)]))
    #    fraud_count.append(df['Class'].sum())

    #Metric calculations
    exp_date = [event for event in target_prediction_from_db['experiment_date'].unique()]
    recall = []
    precision = []
    accuracy = []

    for timestamp in target_prediction_from_db['experiment_date'].unique():
        df = target_prediction_from_db[target_prediction_from_db['experiment_date'] == timestamp ]
        df = df.astype({'actual_class': int, 'predicted_class': int})
        #recall
        recall_df = metrics.recall_score(df['actual_class'],df['predicted_class'])
        recall.append(recall_df)
        #precision
        precision_df = metrics.precision_score(df['actual_class'],df['predicted_class'])
        precision.append(precision_df)
        #accuracy
        accuracy_df = metrics.accuracy_score(df['actual_class'],df['predicted_class'])
        accuracy.append(accuracy_df)


    return html.Div(
    children=[
        html.H1(children="Model Training Monitor"),
        #Start of the raw data visualization
        
        #Model Result Monitoring
        html.H2(children="Model Result Montitoring"),
        dcc.Graph(
            id="model_metrics",
            figure={
                "data": [
                    {"x": exp_date, "y": recall, "type": "line", "name": "Recall"},
                    {"x": exp_date, "y": precision, "type": "line", "name": "Precision"},
                    {"x": exp_date, "y": accuracy, "type": "line", "name": "Accuracy"},
                ],
                "layout": {"title": "Performance Metrics of Model Training over Recent Training Runs"},
            },
        ),
    ])

#function approach
dash_app.layout = serve_layout


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)