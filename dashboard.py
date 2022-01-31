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
from dash import dcc
from dash import html
from dash import dash_table
import plotly
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def yaml_get(filename):
    """
    Loads hyperparameters from a YAML file.
    """
    y = None
    with open(filename) as f:
        y = yaml.load(f.read(),yaml.Loader)
    return y
y = yaml_get(os.path.join(__location__,"dash.yaml"))

COLOR = plotly.colors.qualitative.Alphabet
USE_COLS = list(y['cols'].keys())
X_COL = y['x_col']
THEME = y['theme']
load_figure_template(THEME)
SEP = y['sep']
#Glob path to csv logs
LOG_PATH = glob(y["glob"])
#Number of columns in the csvs
NCOLSDF = None
#Number of last rows to read
NROWSDF = y["tail"]
if len(LOG_PATH)!=0:
    with open(LOG_PATH[0]) as file:
        HEAD = pd.read_csv(io.StringIO(''.join([file.readline() for _ in range(2)])),sep=SEP)
        print(HEAD)
        HEAD = HEAD.columns
else:
    print(f"Glob {y['glob']} doesn't exist.")
    exit()
NCOLSDF = len(USE_COLS)
#Number of columns for the layout
NCOLS = y["display_cols"]
#Number of rows for the layout
NROWS = ceil(NCOLSDF/2)
#Maximum number of points to display at once
MAXPOINTS = y.get("max_points",None)

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

external_stylesheets = [dbc.themes.CYBORG]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H1('DASH'),
        html.P("\t".join(LOG_PATH)),
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

    fig = plotly.subplots.make_subplots(rows=NROWS, cols=NCOLS,subplot_titles=USE_COLS,vertical_spacing=y['vspace'])
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    fig['layout']['template'] = THEME

    for i,colname in enumerate(USE_COLS):
        row, col = RCGEN[i]
        for _name,data in df.items():
            plot_data = data.copy().loc[::len(data)//MAXPOINTS if MAXPOINTS else 1,:]
            fig.append_trace({
                'y': plot_data[colname],
                'x':plot_data[X_COL],
                'name': _name,
                'mode': 'lines',
                'type': 'scatter',
                'legendgroup': _name,
                'showlegend': i == 0,
                'line_color': color_dict[_name]
            }, row, col)
            fig.update_xaxes(title_text=X_COL,row=row,col=col)
            fig.update_yaxes(range=y['cols'][colname],row=row,col=col)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)