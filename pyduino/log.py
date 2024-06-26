import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import io
from glob import glob
from tabulate import tabulate
from collections import OrderedDict
from datetime import datetime

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
config_file = os.path.join(__location__,"config.yaml")

def datetime_from_str(x):
    return datetime.strptime(str(x),"%Y%m%d%H%M%S")

def datetime_to_str(x):
    return x.strftime("%Y%m%d%H%M%S")

def to_markdown_table(data: OrderedDict) -> str:
    """
    Converts the given data into a markdown table format.

    Args:
        data (OrderedDict[OrderedDict]): The data to be converted into a markdown table.

    Returns:
        str: The markdown table representation of the data.
    """
    rows = []
    for rid, rdata in data.items():
        rdata = OrderedDict({"ID": rid, **rdata})
        rows.append(rdata)
    return tabulate(rows, headers="keys", tablefmt="pipe")

def y_to_table(y):
    return tabulate(list(y.items()), tablefmt="pipe")

class Log:
    @property
    def timestamp(self):
        """str: Current date."""
        return datetime.now()
    
    @property
    def prefix(self):
        return os.path.join(self.path,self.start_timestamp)

    def __init__(self,subdir,path="./log",name=None):
        """
        Logs data into jsonls with timestamps.

        Example:
            log_obj = log(['reactor_0','reactor_1'],path='./log',name='experiment_0')

            log/YEAR/MONTH/
            ├─ experiment_0/
            │  ├─ reactor_0.jsonl
            │  ├─ reactor_1.jsonl

        Args:
            subdir (:obj:`list` of :obj:`str`): List of the names for the subdirectories of `path`.
            path (str): Save path for the logs.
            name (str): Name given for this particular instance. If none will name it with the current timestamp.
        """
        self.today = datetime.now()
        self.path = os.path.join(path, self.today.strftime("%Y"), self.today.strftime("%m"))
        self.start_timestamp = datetime_to_str(self.timestamp) if name is None else name
        self.log_name = name
        Path(os.path.join(self.path,self.start_timestamp)).mkdir(parents=True,exist_ok=True)
        if isinstance(subdir,str):
            self.subdir = list(map(os.path.basename,glob(os.path.join(self.prefix,subdir))))
        elif isinstance(subdir,list):
            self.subdir = subdir
        else:
            raise ValueError("Invalid type for subdir. Must be either a list of strings or a glob string.")
        self.subdir = list(map(lambda x: str(x)+".jsonl" if len(os.path.splitext(str(x))[1])==0 else str(x),self.subdir))
        self.first_timestamp = None
        self.data_frames = {}

        self.paths = list(map(lambda x: os.path.join(self.prefix,x),self.subdir))
        self.log_name = name
        Path(os.path.join(self.path,self.start_timestamp)).mkdir(parents=True,exist_ok=True)
        if isinstance(subdir,str):
            self.subdir = list(map(os.path.basename,glob(os.path.join(self.prefix,subdir))))
        elif isinstance(subdir,list):
            self.subdir = subdir
        else:
            raise ValueError("Invalid type for subdir. Must be either a list of strings or a glob string.")
        self.subdir = list(map(lambda x: str(x)+".jsonl" if len(os.path.splitext(str(x))[1])==0 else str(x),self.subdir))
        self.first_timestamp = None
        self.data_frames = {}

        self.paths = list(map(lambda x: os.path.join(self.prefix,x),self.subdir))

    def backup_config_file(self):
        filename = os.path.join(self.path,self.start_timestamp,f"{self.start_timestamp.replace('/','-')}.yaml")
        if not os.path.exists(filename):
            with open(config_file) as cfile, open(filename,'w') as wfile:
                wfile.write(cfile.read())

    def log_rows(self,rows,subdir,add_timestamp=True,tags=None):
        """
        Logs rows into jsonl format.

        Args:
            rows (:obj:`list` of :obj:`dict`): List of dictionary-encoded rows or pandas dataframe.
            subdir (str): Subdirectory name. Intended to be an element of `self.subdir`.
            add_timestamp (bool,optional): Whether or not to include a timestamp column.
            tags (:obj:`dict` of :obj:`str`): Dictionary of strings to be inserted as constant columns.
        """
        t = self.timestamp
        path = os.path.join(self.path,self.start_timestamp,f"{subdir}.jsonl")

        df = pd.DataFrame()
        if isinstance(rows,list):
            df = pd.DataFrame(rows)
        elif isinstance(rows,pd.DataFrame):
            df = rows.copy()
        
        if add_timestamp:
            df.loc[:,"log_timestamp"] = datetime_to_str(t)
        if os.path.exists(path):
            if self.first_timestamp is None:
                with open(path) as file:
                    head = pd.read_json(io.StringIO(file.readline()+file.readline()), orient="records", lines=True)
                    self.first_timestamp = datetime_from_str(head.log_timestamp[0])
        else:
            self.first_timestamp = t
        df.loc[:,"elapsed_time_hours"] = (t - self.first_timestamp).total_seconds()/3600.0

        #Inserting constant values
        if tags is not None:
            for key,value in tags.items():
                df.loc[:,key] = value

        with open(path, mode="a") as log_file:
            log_file.write(df.to_json(orient="records", lines=True))

        return df
    def log_many_rows(self,data,**kwargs):
        """
        Logs rows into jsonl format.

        Args:
            data (:obj:`dict` of :obj:`dict`): Dictionary encoded data frame.
            **kwargs: Additional arguments passed to `self.log_rows`.
        """
        self.data_frames = {}
        for _id,row in data.items():
            df = self.log_rows(rows=[row],subdir=_id,**kwargs)
            self.data_frames[_id] = df
        self.data_frames = pd.concat(list(self.data_frames.values()))
    
    def log_optimal(self,column,maximum=True,**kwargs):
        """
        Logs optima of all rows into a single file.
        """
        i=self.data_frames.loc[:,column].astype(float).argmax() if maximum else self.data_frames.loc[:,column].astype(float).argmin()
        self.df_opt = self.data_frames.iloc[i,:]
        self.log_rows(rows=[self.df_opt.to_dict()],subdir='opt',**kwargs)
    
    def log_average(self, cols: list, **kwargs):
        """
        Calculate the average values of specified columns in the data frames and log the results.

        Parameters:
        - cols (list): A list of column names to calculate the average for.
        - **kwargs: Additional keyword arguments to customize the logging process.
        """
        df = self.data_frames.copy()
        df.loc[:, cols] = df.loc[:, cols].astype(float)
        df.elapsed_time_hours = df.elapsed_time_hours.round(decimals=2)
        self.df_avg = df.loc[:, cols + ['elapsed_time_hours']].groupby("elapsed_time_hours").mean().reset_index()
        self.log_rows(rows=self.df_avg, subdir='avg',  **kwargs)

    def cache_data(self,rows,path="./cache.jsonl",**kwargs):
        """
        Dumps rows into a single jsonl.

        Args:
            rows (:obj:`list` of :obj:`dict`): List of dictionary-encoded rows.
            path (str): Path to the jsonl file.
        """
        pd.DataFrame(rows).T.to_json(path, orient="records", lines=True, **kwargs)

    def transpose(self,columns,destination,skip=1,**kwargs):
        """
        Maps reactor jsonl to column jsonls with columns given by columns.

        Args:
            columns (:obj:list of :obj:str): List of columns to extract.
            destination (str): Destination path. Creates directories as needed and overwrites any existing files.

            skip (int, optional): How many rows to jump while reading the input files. Defaults to 1.
        """
        dfs = []
        for file in self.paths:
            df = pd.read_json(file, orient="records", lines=True, **kwargs)
            df['FILE'] = file
            dfs.append(df.iloc[::skip,:])
        df = pd.concat(dfs)

        for column in columns:
            Path(destination).mkdir(parents=True,exist_ok=True)
            df.loc[:,['ID','FILE',column,'elapsed_time_hours']].to_json(os.path.join(destination,f"{column}.jsonl"), orient="records", lines=True)


class LogAggregator:
    def __init__(self,log_paths,timestamp_col="log_timestamp",elapsed_time_col="elapsed_time_hours"):
        """
        Merges logs from various experiments into a single file for each bioreactor.

        Args:
            log_paths (:obj:list of :obj:str): List of glob strings pointing at the input files for each experiment.
            timestamp_col (str, optional): Column to use as timestamp. Defaults to "log_timestamp".
            elapsed_time_col (str, optional): Columns to use as 'elapsed time'. Defaults to "elapsed_time_hours".
        """
        self.glob_list = log_paths
        self.timestamp_col = timestamp_col
        self.elapsed_time_col = elapsed_time_col
    def agg(self,destination,skip=1,**kwargs):
        """
        Aggregator

        Args:
            destination (str): Destination path. Creates directories as needed and overwrites any existing files.
            skip (int, optional): How many rows to jump while reading the input files. Defaults to 1.
        """
        dfs = {}
        for path in self.glob_list:
            for file in glob(path):
                basename = os.path.basename(file)
                df = pd.read_json(file, orient="records", lines=True, dtype={self.elapsed_time_col:float},**kwargs)
                df = df.iloc[::skip,:]
                df['FILE'] = file
                if dfs.get(basename,None) is not None:
                    top_timestamp = datetime_from_str(df.head(1)[self.timestamp_col].iloc[0])
                    bottom_timestamp = datetime_from_str(dfs.get(basename).tail(1)[self.timestamp_col].iloc[0])
                    bottom_elapsed_time = dfs.get(basename).tail(1)[self.elapsed_time_col].iloc[0]
                    deltaT = (top_timestamp - bottom_timestamp).total_seconds()/3600.0
                    print("DeltaT",deltaT)
                    print(df[self.elapsed_time_col].head())
                    df[self.elapsed_time_col] = df[self.elapsed_time_col] + deltaT + bottom_elapsed_time
                    print(df[self.elapsed_time_col].head())
                    dfs[basename] = pd.concat([dfs[basename],df])
                else:
                    dfs[basename] = df
        for filename, df in dfs.items():
            Path(destination).mkdir(parents=True,exist_ok=True)
            path = os.path.join(destination,filename)
            df.to_json(path, orient="records", lines=True)

