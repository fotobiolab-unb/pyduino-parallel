import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import io
from glob import glob

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
config_file = os.path.join(__location__,"config.yaml")

def datetime_from_str(x):
    return datetime.strptime(str(x),"%Y%m%d%H%M%S")

def datetime_to_str(x):
    return x.strftime("%Y%m%d%H%M%S")

class log:
    @property
    def timestamp(self):
        """str: Current date."""
        return datetime.now()
    
    @property
    def prefix(self):
        return os.path.join(self.path,self.start_timestamp)

    def __init__(self,subdir,path="./log",name=None):
        """
        Logs data into csvs with timestamps.

        Example:
            log_obj = log(['reactor_0','reactor_1'],path='./log',name='experiment_0')

            log/
            ├─ experiment_0/
            │  ├─ reactor_0.csv
            │  ├─ reactor_1.csv

        Args:
            subdir (:obj:`list` of :obj:`str`): List of the names for the subdirectories of `path`.
            path (str): Save path for the logs.
            name (str): Name given for this particular instance. If none will name it with the current timestamp.
        """
        self.path = path
        self.start_timestamp = datetime_to_str(self.timestamp) if name is None else name
        self.log_name = name
        Path(os.path.join(self.path,self.start_timestamp)).mkdir(parents=True,exist_ok=True)
        if isinstance(subdir,str):
            self.subdir = list(map(os.path.basename,glob(os.path.join(self.prefix,subdir))))
        elif isinstance(subdir,list):
            self.subdir = subdir
        else:
            raise ValueError("Invalid type for subdir. Must be either a list of strings or a glob string.")
        self.subdir = list(map(lambda x: str(x)+".csv" if len(os.path.splitext(str(x))[1])==0 else str(x),self.subdir))
        self.first_timestamp = None
        self.data_frames = {}

        self.paths = list(map(lambda x: os.path.join(self.prefix,x),self.subdir))

        with open(config_file) as cfile, open(os.path.join(self.path,self.start_timestamp,f"{self.start_timestamp}.yaml"),'w') as wfile:
            wfile.write(cfile.read())


    def log_rows(self,rows,subdir,add_timestamp=True,**kwargs):
        """
        Logs rows into csv format.

        Args:
            rows (:obj:`list` of :obj:`dict`): List of dictionary-encoded rows.
            subdir (str): Subdirectory name. Intended to be an element of `self.subdir`.
            add_timestamp (bool,optional): Whether or not to include a timestamp column.
            **kwargs: Additional arguments passed to `pandas.to_csv`.
        """
        t = self.timestamp
        path = os.path.join(self.path,self.start_timestamp,f"{subdir}.csv")
        df = pd.DataFrame(rows)
        if add_timestamp:
            df.loc[:,"log_timestamp"] = datetime_to_str(t)
        if os.path.exists(path):
            if self.first_timestamp is None:
                with open(path) as file:
                    head = pd.read_csv(io.StringIO(file.readline()+file.readline()),index_col=False,**kwargs)
                    self.first_timestamp = datetime_from_str(head.log_timestamp[0])
        else:
            self.first_timestamp = t
        df.loc[:,"elapsed_time_hours"] = (t - self.first_timestamp).total_seconds()/3600.0
        df.to_csv(
            path,
            mode="a",
            header=not os.path.exists(path),
            index=False,
            **kwargs
            )
        return df
    def log_many_rows(self,data,**kwargs):
        """
        Logs rows into csv format.

        Args:
            data (:obj:`dict` of :obj:`dict`): Dictionary encoded data frame.
            **kwargs: Additional arguments passed to `self.log_rows`.
        """
        self.data_frames = {}
        for _id,row in data.items():
            df = self.log_rows(rows=[row],subdir=_id,sep='\t',**kwargs)
            self.data_frames[_id] = df
        self.data_frames = pd.concat(list(self.data_frames.values()))
    
    def log_optimal(self,column,maximum=True,**kwargs):
        """
        Logs optima of all rows into a single file.
        """
        i=self.data_frames.loc[:,column].argmax() if maximum else self.data_frames.loc[:,column].argmin()
        self.df_opt = self.data_frames.iloc[i,:]
        self.log_rows(rows=[self.df_opt.to_dict()],subdir='opt',sep='\t',**kwargs)


    def cache_data(self,rows,path="./cache.csv",**kwargs):
        """
        Dumps rows into a single csv.

        Args:
            rows (:obj:`list` of :obj:`dict`): List of dictionary-encoded rows.
            path (str): Path to the csv file.
        """
        pd.DataFrame(rows).T.to_csv(path,**kwargs)

    def transpose(self,columns,destination,sep='\t',skip=1,**kwargs):
        """
        Maps reactor csv to column csvs with columns given by columns.

        Args:
            columns (:obj:list of :obj:str): List of columns to extract.
            destination (str): Destination path. Creates directories as needed and overwrites any existing files.
            sep (str, optional): Column separator. Defaults to '\t'.
            skip (int, optional): How many rows to jump while reading the input files. Defaults to 1.
        """
        dfs = []
        for file in self.paths:
            df = pd.read_csv(file,index_col=False,sep=sep,**kwargs)
            df['FILE'] = file
            dfs.append(df.iloc[::skip,:])
        df = pd.concat(dfs)

        for column in columns:
            Path(destination).mkdir(parents=True,exist_ok=True)
            df.loc[:,['ID','FILE',column,'elapsed_time_hours']].to_csv(os.path.join(destination,f"{column}.csv"),sep=sep)


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
    def agg(self,destination,skip=1,sep='\t',**kwargs):
        """
        Aggregator

        Args:
            destination (str): Destination path. Creates directories as needed and overwrites any existing files.
            skip (int, optional): How many rows to jump while reading the input files. Defaults to 1.
            sep (str, optional): Column separator. Defaults to '\t'.
        """
        dfs = {}
        for path in self.glob_list:
            for file in glob(path):
                basename = os.path.basename(file)
                df = pd.read_csv(file,index_col=False,sep=sep,dtype={self.elapsed_time_col:float},**kwargs)
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
            df.to_csv(path,sep=sep,index=False)

