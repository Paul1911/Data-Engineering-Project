import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import flask

# start own
import pandas as pd
from sqlalchemy import create_engine
import ml_pipeline_config_test as configurations #todo: find a better solution instead of this dupe file 

db_engine = configurations.params["db_engine"]
db_schema = configurations.params["db_schema"]
table_raw = configurations.params["db_raw_table"] 
table_results = configurations.params["db_results_table"]

engine = create_engine(db_engine)
results_from_db = pd.read_sql_table('results', con = engine)
#results_from_db.to_csv('/app/results_from_db.csv', sep=',', index=False)


#end own


app= flask.Flask(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#dash_app = dash.Dash(__name__, server=app, external_stylesheets=[dbc.themes.BOOTSTRAP])
dash_app = dash.Dash(__name__, server=app, external_stylesheets=external_stylesheets)
dash_app.config.suppress_callback_exceptions = True

dash_app.layout = html.Div(
    children=[
        html.H1(children="Model Training Monitor"),
        html.Div(children="""Displaying the PostgreSQL-data in a DASH WebApp"""),
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
            id="example-graph_2",
            figure={
                "data": [
                    {"x": [1, 2, 3], "y": [4, 1, 2], "type": "line", "name": "$F_1$ Example"},
                    {
                        "x": [1, 2, 3],
                        "y": [2, 4, results_from_db.iloc[-1,-1]],
                        "type": "line",
                        "name": "Accuracy",
                    },
                ],
                "layout": {"title": "Performance Metrics of Model Training over Recent Training Runs"},
            },
        ),
    ]
)



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)