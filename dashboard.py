import datetime
import enum
import pandas as pd
from glob import glob
import os
from math import ceil
from itertools import product
import tailer
import io
import yaml

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
from dash.dependencies import Input, Output

def yaml_get(filename):
    """
    Loads hyperparameters from a YAML file.
    """
    y = None
    with open(filename) as f:
        y = yaml.load(f.read(),yaml.Loader)
    return y
y = yaml_get("dash.yaml")

COLOR = plotly.colors.qualitative.Alphabet
USE_COLS = y['cols']
THEME = y['theme']
SEP = y['sep']
#Glob path to csv logs
LOG_PATH = glob(y["glob"])
#Number of columns in the csvs
NCOLSDF = None
#Number of last rows to read
NROWSDF = y["tail"]
with open(LOG_PATH[0]) as file:
    HEAD = pd.read_csv(io.StringIO(''.join([file.readline() for _ in range(2)])),sep=SEP)
    print(HEAD)
    HEAD = HEAD.columns
NCOLSDF = len(USE_COLS)
#Number of columns for the layout
NCOLS = y["display_cols"]
#Number of rows for the layout
NROWS = ceil(NCOLSDF/2)

RCGEN = list(product(range(1,NROWS+1),range(1,NCOLS+1)))

def tail(fname,n,**kwargs):
    """
    Reads n last rows of `fname`.
    Args:
        fname (str): Path to read from.
        n (int): Last n rows to read.
    """ 
    with open(fname) as file:
        last_lines = tailer.tail(file, n)
    return pd.read_csv(io.StringIO('\n'.join(last_lines)),**kwargs)

color_dict = {os.path.basename(_name): COLOR[i] for i,_name in enumerate(LOG_PATH)}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('DASH'),
        dcc.Graph(id='live-update-graph',style={'height': y['subplot_height']*len(USE_COLS)}),
        dcc.Interval(
            id='interval-component',
            interval=y['update_time'], # in milliseconds
            n_intervals=0
        )
    ])
)

df = {}
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    global df
    if not y['cumulative']:
        df = {}
    for gpath in LOG_PATH:
        gbase = os.path.basename(gpath)
        if gbase in df.keys():
            df[gbase] = pd.merge(df[gbase],tail(gpath,NROWSDF,sep=SEP),'outer')
        else:
            df[gbase] = tail(gpath,NROWSDF,sep=SEP)
            df[gbase].columns = HEAD

    fig = plotly.tools.make_subplots(rows=NROWS, cols=NCOLS,subplot_titles=USE_COLS)
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    fig['layout']['template'] = THEME

    for i,colname in enumerate(USE_COLS):
        row, col = RCGEN[i]
        for _name,data in df.items():
            fig.append_trace({
                #'x': data['time'],
                'y': data[colname],
                'name': _name,
                'mode': 'lines+markers',
                'type': 'scatter',
                'legendgroup': _name,
                'showlegend': i == 0,
                'line_color': color_dict[_name]
            }, row, col)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)