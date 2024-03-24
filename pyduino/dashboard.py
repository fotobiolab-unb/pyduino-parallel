import pandas as pd
from glob import glob
import os
from math import ceil
from itertools import product
import tailer
import io

import dash
from dash import dcc
from dash import html
from dash import dash_table
import plotly.graph_objects as go
import plotly
import plotly.express as px
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import yaml
from pyduino.pyduino2 import CACHEPATH
from pyduino.utils import yaml_get

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

y = yaml_get(os.path.join(__location__,"config.yaml"))['dash']
RELEVANT_PARAMETERS = yaml_get(os.path.join(__location__,"config.yaml"))['system']['relevant_parameters']
IRRADIANCE = yaml_get(os.path.join(__location__,"config.yaml"))['system']['irradiance']
COLOR = plotly.colors.qualitative.Alphabet
PLOTS = list(y['plot'])
SUBPLOT_NAMES = list(map(lambda x: x.get('name',"nameless"),PLOTS))
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
        HEAD = pd.read_csv(io.StringIO(''.join([file.readline() for _ in range(1)])),sep=SEP)
        print(HEAD)
        HEAD = HEAD.columns
else:
    raise ValueError(f"Glob {y['glob']} doesn't exist.")
NCOLSDF = len(PLOTS)
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
        html.P("Glob: "+"\t".join(y["glob"])),
        dcc.Graph(id='matrix',style={'margin':15}),
        dcc.Graph(id='live-update-graph',style={'height': y['subplot_height']*NCOLSDF, 'margin':15}),
        dcc.Interval(
            id='interval-component',
            interval=y['update_time'], # in milliseconds
            n_intervals=0
        )
    ])
)

df = {}
@app.callback(Output('matrix','figure'),Input('interval-component','n_intervals'))
def get_matrix(n):
    cache_df = pd.read_csv(CACHEPATH,sep='\t',index_col='ID')
    cache_df.columns = cache_df.columns.str.lower()
    cache_df = cache_df.loc[:,RELEVANT_PARAMETERS]
    cache_df = cache_df * pd.Series(IRRADIANCE)/100
    fig = go.Figure(data=go.Heatmap(
        z=cache_df.to_numpy(),
        x=cache_df.columns,
        y=cache_df.index.astype(str).to_list(),
        colorscale='Inferno'
    ))
    fig['layout']['template'] = THEME
    fig['layout']['title'] = "Computed Parameters"
    fig.data[0].update(zmin=0,zmax=2)
    fig['layout']['font']['size']=20
    fig['layout']['font']['family']="monospace"
    return fig

@app.callback(Output('live-update-graph', 'figure'),Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    global df
    if not y['cumulative']:
        df = {}
    for gpath in LOG_PATH:
        gbase = os.path.basename(gpath)
        if gbase in df.keys():
            df_tail = tail(gpath,NROWSDF,sep=SEP,header=None)
            df_tail.columns = df[gbase].columns
            df[gbase] = pd.concat([df[gbase],df_tail],ignore_index=True).copy()
        else:
            df[gbase] = pd.read_csv(gpath,sep=SEP)#tail(gpath,NROWSDF,sep=SEP)

    fig = plotly.subplots.make_subplots(rows=NROWS, cols=NCOLS,subplot_titles=SUBPLOT_NAMES,vertical_spacing=y['vspace'])
    fig['layout']['legend'] = {'y':1.08,'orientation':'h'}
    fig['layout']['template'] = THEME

    for i,plot in enumerate(PLOTS):
        row, col = RCGEN[i]
        fig['layout'][f"yaxis{i+1}"]['title'] = plot['name']
        for colname,options in plot['cols'].items():
            if options is None:
                options = {}
            for _name,data in df.items():
                cap = len(data)//MAXPOINTS if MAXPOINTS else 1
                cap = cap if cap!=0 else 1
                plot_data = data.copy().loc[::cap,:]

                if colname in plot_data.columns:

                    if options.get('growing_only',False) and ("growth_state" in plot_data.columns):
                        plot_data = plot_data.loc[plot_data.growth_state=="growing",:]

                    y_data = plot_data[colname]
                    if options.get('positive_only',False):
                        y_data = y_data.mask(y_data.lt(0)).ffill().fillna(0).convert_dtypes()

                    if X_COL in plot_data.columns:
                        fig.add_trace({
                            'y': y_data,
                            'x':plot_data[X_COL],
                            'name': f"{_name}({cap})",
                            'mode': options.get('mode','lines'),
                            'type': 'scatter',
                            'legendgroup': _name,
                            'showlegend': colname=='power',
                            'line_color': color_dict[_name],
                            'line': {'dash':options.get('dash',None)}
                        }, row, col)

    fig['layout']['font']['size']=15
    fig['layout']['font']['family']="monospace"

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', dev_tools_silence_routes_logging = False)
