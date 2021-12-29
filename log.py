import pandas as pd
import os
import path
from pathlib import Path
from datetime import datetime

class log:
    @property
    def timestamp(self):
        """str: Current date."""
        return datetime.now().strftime("%Y%m%d%H%M%S")

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
        self.start_timestamp = self.timestamp if name is None else name
        self.log_name = name
        self.subdir = subdir
        self.path = path

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
        path = os.path.join(self.path,self.start_timestamp,f"{subdir}.csv")
        df = pd.DataFrame(rows)
        if add_timestamp:
            df.loc[:,"log_timestamp"] = self.timestamp
        df.to_csv(
            path,
            mode="a",
            header=not os.path.exists(path),
            **kwargs
            )
    def cache_data(self,rows,path="./cache.csv",**kwargs):
        """
        Dumps rows into a single csv.

        Args:
            rows (:obj:`list` of :obj:`dict`): List of dictionary-encoded rows.
            path (str): Path to the csv file.
        """
        pd.DataFrame(rows).T.to_csv(path,**kwargs)