import pandas as pd
import os
import path
from pathlib import Path
from datetime import datetime
import io

def datetime_from_str(x):
    return datetime.strptime(str(x),"%Y%m%d%H%M%S")

def datetime_to_str(x):
    return x.strftime("%Y%m%d%H%M%S")

class log:
    @property
    def timestamp(self):
        """str: Current date."""
        return datetime.now()

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
        self.start_timestamp = datetime_to_str(self.timestamp) if name is None else name
        self.log_name = name
        self.subdir = subdir
        self.path = path
        self.first_timestamp = None

        for sbd in subdir:
            Path(os.path.join(self.path,self.start_timestamp)).mkdir(parents=True,exist_ok=True)

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
                    head = pd.read_csv(io.StringIO(file.readline()+file.readline()),**kwargs)
                    self.first_timestamp = datetime_from_str(head.log_timestamp[0])
        else:
            self.first_timestamp = t
        df.loc[:,"elapsed_time_hours"] = (t - self.first_timestamp).total_seconds()/3600.0
        df.to_csv(
            path,
            mode="a",
            header=not os.path.exists(path),
            **kwargs
            )
    def log_many_rows(self,data,**kwargs):
        """
        Logs rows into csv format.

        Args:
            data (:obj:`dict` of :obj:`dict`): Dictionary encoded data frame.
            **kwargs: Additional arguments passed to `self.log_rows`.
        """
        for _id,row in data.items():
            self.log_rows(rows=[row],subdir=_id,sep='\t',index=False,**kwargs)
    def cache_data(self,rows,path="./cache.csv",**kwargs):
        """
        Dumps rows into a single csv.

        Args:
            rows (:obj:`list` of :obj:`dict`): List of dictionary-encoded rows.
            path (str): Path to the csv file.
        """
        pd.DataFrame(rows).T.to_csv(path,**kwargs)