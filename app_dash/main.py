import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import flask
from sklearn import metrics 

import pandas as pd
from sqlalchemy import create_engine
import ml_pipeline_config_test as configurations #todo: find a better solution instead of this dupe file 

db_engine = configurations.params["db_engine"]
db_schema = configurations.params["db_schema"]
table_raw = configurations.params["db_raw_table"] 
table_results = configurations.params["db_results_table"]
table_target_prediction = configurations.params["db_target_prediction_table"] 

engine = create_engine(db_engine)


app= flask.Flask(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#dash_app = dash.Dash(__name__, server=app, external_stylesheets=[dbc.themes.BOOTSTRAP])
dash_app = dash.Dash(__name__, server=app, external_stylesheets=external_stylesheets)
dash_app.config.suppress_callback_exceptions = True

#function approach 
def serve_layout():
    #Read tables
    results_from_db = pd.read_sql_table('results', con = engine)
    target_prediction_from_db = pd.read_sql_table('target_prediction', con = engine)

    #Metric calculations
    exp_date = [event for event in target_prediction_from_db['experiment_date'].unique()]
    recall = []
    precision = []
    accuracy = []

    #def metric_calculations():
    for timestamp in target_prediction_from_db['experiment_date'].unique():
        df = target_prediction_from_db[target_prediction_from_db['experiment_date'] == timestamp ]
        df = df.astype({'actual_class': int, 'predicted_class': int})
        print(df.dtypes)
        print(df.head())
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
#dash_app.layout = html.Div( #traditional approach
    children=[
        html.H1(children="Model Training Monitor"),
        html.H2(children="Raw Data Quality Montitoring"),
        html.H2(children="Model Result Montitoring"),
        html.Div(children="""Displaying the PostgreSQL-data in a Dash WebApp"""),
        dcc.Graph(
            id="example-graph",
            figure={
                "data": [
                    {"x": [1, 2, 3], "y": [4, 15, 2], "type": "bar", "name": "Like"},
                    {
                        "x": [1, 2, 3],
                        "y": [2, 4, results_from_db.iloc[-1,-1]],
                        "type": "bar",
                        "name": "Comment",
                    },
                ],
                "layout": {"title": "Like Vs Comment Dash Visualization"},
            },
        ),
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
    ]
)

#function approach
dash_app.layout = serve_layout

""" @dash_app.callback(dash.dependencies.Output('model_metrics', 'figure'),
#    [dash.dependencies.Input('refresh_data', 'n_clicks')])
    [dash.dependencies.Input('interval-component', 'n_intervals')])

def update_model_metrics():
    target_prediction_from_db = pd.read_sql_table('target_prediction', con = engine)

    #Metric calculations
    exp_date = [event for event in target_prediction_from_db['experiment_date'].unique()]
    recall = []
    precision = []
    accuracy = []
    roc_auc = []

    for timestamp in target_prediction_from_db['experiment_date'].unique():
        df = target_prediction_from_db[target_prediction_from_db['experiment_date'] == timestamp ]
        df = df.astype({'actual_class': int, 'predicted_class': int})
        print(df.dtypes)
        print(df.head())
        #recall
        recall_df = metrics.recall_score(df['actual_class'],df['predicted_class'])
        recall.append(recall_df)
        #precision
        precision_df = metrics.precision_score(df['actual_class'],df['predicted_class'])
        precision.append(precision_df)
        #accuracy
        accuracy_df = metrics.accuracy_score(df['actual_class'],df['predicted_class'])
        accuracy.append(accuracy_df)
        #roc_auc
        roc_auc_df = metrics.roc_auc_score(df['actual_class'],df['predicted_class'])
        roc_auc.append(roc_auc_df)

        #Create Figure
        figure={
                "data": [
                    {"x": exp_date, "y": recall, "type": "line", "name": "Recall"},
                    {"x": exp_date, "y": precision, "type": "line", "name": "Precision"},
                    {"x": exp_date, "y": accuracy, "type": "line", "name": "Accuracy"},
                ],
                "layout": {"title": "Performance Metrics of Model Training over Recent Training Runs"},
            },

        # Create plotly figure
    return figure """


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)