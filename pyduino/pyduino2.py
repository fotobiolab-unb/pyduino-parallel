from typing import Union
import logging
from time import sleep, time
import pandas as pd
from collections import OrderedDict
from pyduino.log import Log
import pandas as pd
from multiprocessing import Pool, Process
from functools import partial
import os
from pyduino.utils import bcolors, get_servers, get_meta
import requests
from urllib.parse import urljoin
import logging
from datetime import datetime
from pyduino.paths import PATHS

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
logging.basicConfig(
    filename='pyduino.log',
    filemode='w',
    level=PATHS.SYSTEM_PARAMETERS.get('log_level', logging.INFO),
)


STEP = 1/8
HEADER_DELAY = 5
COLN = 48 #Number of columns to parse from Arduino (used for sanity tests)
CACHEPATH = "cache.csv"

#From https://stackoverflow.com/a/312464/6451772
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class Reactor:
    """
    Master of HTTP Server to Serial handler. 
    """

    def __init__(self, url):
        self.connected = False
        self.url = url
        self.meta = get_meta(url)
        self.id = self.meta["id"]

    def http_get(self,route):
        return requests.get(urljoin(self.url,route))
    
    def http_post(self,route,command,await_response):
        return requests.post(urljoin(self.url,route),json={
            "command": command,
            "await_response": await_response
        })

    def connect(self):
        """
        Starts connection
        """
        resp = self.http_get("connect")
        if resp.ok:
            self.connected = True
        return resp.ok
    
    def reboot(self,retry_time=5):
        try:
            resp = self.http_get("reboot")
        except requests.ConnectionError:
            self.connected = False
        print(bcolors.WARNING,"Rebooting reactor",self.id,"at",self.url,bcolors.ENDC)
        while not self.connected:
            try:
                self.connect()
            except requests.ConnectionError:
                print(bcolors.WARNING,"Waiting for reactor",self.id,bcolors.ENDC)
                sleep(retry_time)

    def reset(self):
        """
        Resets connection.
        """
        resp = self.http_get("reset")
        return resp.ok

    def close(self):
        """
        Sends 'fim'.
        """
        self.http_post("send","fim",False,0)

    def send(self, msg):
        """
        Sends command and awaits for a response
        """
        resp = self.http_post("send",msg,True)
        return resp.json()["response"]

    def _send(self, msg):
        """
        Sends command and doesn't await for a response
        """
        resp = self.http_post("send",msg,False)
        return resp.ok
    
    def set_in_chunks(self,params,chunksize=4):
        """
        Sets params into chunks.
        """
        ch = chunks(params,chunksize)
        for chunk in ch:
            self.set(dict(chunk))
            sleep(2)
    
    def __repr__(self):
        return f"{bcolors.OKCYAN}<Reactor {self.id} at {self.meta['hostname']}({self.url})>{bcolors.ENDC}"

    def set(self, data=None, **kwargs):
        """
        Reactor.set({"440": 50, "brilho": 100})
        """
        data = {**(data or {}), **kwargs}
        args = ",".join(f'{k},{v}' for k, v in data.items())
        cmd = f"set({args})"
        self._send(cmd)
    
    def horacerta(self):
        """
        Synchronizes Arduino clock with the client computer.
        """
        now = datetime.now()
        logging.info(f"Set clock on {self.meta['hostname']}")
        self.set_in_chunks([["ano",now.year],["mes",now.month],["dia",now.day],["hora",now.hour],["minuto",now.minute]],chunksize=2)
        sleep(2)
        logging.debug(f"Clock set on {self.meta['hostname']}")
        self._send("horacerta")


def send_wrapper(reactor,command,await_response):
    id, reactor = reactor
    if await_response:
        return (id,reactor.send(command))
    else:
        return (id,reactor._send(command))

def reboot_wrapper(reactor):
    id,reactor = reactor
    reactor.reboot(retry_time=PATHS.SYSTEM_PARAMETERS["reboot_wait_time"])
    return True

def set_in_chunks(X):
    reactor,row,chunksize = X
    reactor.set_in_chunks(list(row.items()),chunksize)

class ReactorManager:
    """
    A class that manages multiple reactors.

    Attributes:
        pinged (bool): Indicates if the reactors have been pinged.
        network (str): The network address of the reactors.
        port (int): The port number of the reactors.
        exclude (list): A list of reactors to exclude.
        reactors (dict): A dictionary of reactor objects.
        servers (dict): A dictionary of server addresses.
        header (list): A list of header values.
        payload (dict): A dictionary of payload values.
        connected (bool): Indicates if the reactors are connected.
        log (log): A log object for logging data.

    Methods:
        __init__(self, include: dict = None): Initializes the ReactorManager object.
        ids(self) -> list: Returns a list of reactor IDs.
        send(self, command, await_response=True, **kwargs): Sends a command to the reactors.
        send_parallel(self, command, delay, await_response=True): Sends a command to the reactors in parallel.
        set(self, data=None, **kwargs): Sets data on the reactors.
        get(self, key=None): Gets data from the reactors.
        connect(self): Connects to the reactors.
        reset(self): Resets the reactors.
        reboot(self): Reboots the reactors.
        horacerta(self): Updates Arduino clocks with the clock of the current system.
        log_init(self, **kwargs): Creates log directories for each Arduino.
        dados(self, save_cache=True): Gets data from the Arduinos.
        log_dados(self, save_cache=True): Logs output of `dados` in CSV format.
        set_preset_state(self, path="preset_state.csv", sep="\t", chunksize=4, params=PATHS.REACTOR_PARAMETERS, **kwargs): Prepares Arduinos with preset parameters from a CSV file.
        calibrate(self, deltaT=120, dir="calibrate"): Runs `curva` and dumps the result into txts.
    """
    
    pinged = False
    def __init__(self, include: dict = None):
        """
        Initializes the ReactorManager object.

        Args:
            include (dict): A dictionary of reactor IDs and their corresponding server addresses.
        """
        self.network = PATHS.SLAVE_PARAMETERS["network"]
        self.port = PATHS.SLAVE_PARAMETERS["port"]
        self.exclude = PATHS.SLAVE_PARAMETERS.get("exclude", None)
        self.reactors = {}

        if include is None:
            self.servers = get_servers(self.network,self.port,self.exclude)
        else:
            self.servers = include

        for id, host in self.servers.items():
            self.reactors[id] = Reactor(host)
        
        logging.info("Connection completed")

        #Info
        print("\n".join(map(lambda x: f"Reactor {bcolors.OKCYAN}{x.id}{bcolors.ENDC} at {bcolors.OKCYAN}{x.meta['hostname']}({x.url}){bcolors.ENDC}",self.reactors.values())))
        self.header = None
        self.payload = None

        if PATHS.SYSTEM_PARAMETERS.get("sync_clocks", True):
            self.horacerta()
        self.ids = sorted(list(self.reactors.keys()))

    def send(self,command,await_response=True,**kwargs):
        out = {}
        for k,r in self.reactors.items():
            if await_response:
                out[k] = r.send(command,**kwargs)
            else:
                r._send(command)
        return out
    
    def send_parallel(self,command,await_response=True):
        with Pool(7) as p:
            out = p.map(partial(send_wrapper,command=command,await_response=await_response),list(self.reactors.items()))
        return dict(out)

    def set(self, data=None, **kwargs):
        for k,r in self.reactors.items():
            r.set(data=data, **kwargs)
    def get(self,key=None):
        for k,r in self.reactors.items():
            r.get(key=key)
    def connect(self):
        for k,r in self.reactors.items():
            r.connect()
    def reset(self):
        for k,r in self.reactors.items():
            r.reset()
        self.connected = True
    def reboot(self):
        self.connected = False
        with Pool(7) as p:
            response = p.map(reboot_wrapper,list(self.reactors.items()))
        self.connected = True
    def horacerta(self):
        """
        Updates Arduino clocks with clock of current system.
        """
        print("[INFO]", f"{bcolors.BOLD}Syncronizing clocks{bcolors.ENDC}")
        for k,r in self.reactors.items():
            r.horacerta()

    #Logging

    def log_init(self,**kwargs):
        """
        Creates log directories for each Arduino.

        Args:
            name (str): Name of the subdirectory in the log folder where the files will be saved.
        """
        self.log = Log(subdir=list(self.reactors.keys()),**kwargs)
        print(f"Log will be saved on: {bcolors.OKGREEN}{self.log.prefix}{bcolors.ENDC}")
    
    @property
    def brilho(self):
        """
        Convenience method to get brilho from reactors.
        """
        out = self.send_parallel(f"get({self.brilho_param.lower()})")
        out = {k: float(v.strip()) for k,v in out.items()}
        return out

    def dados(self,save_cache=True):
        """
        Get data from Arduinos.

        Args:
            save_cache (bool): Whether or not so save a cache file with the last reading with `log.log.cache_data`.
        """

        if self.header is None:
            self.header = list(self.reactors.values())[0].send("cabecalho").split(" ")

        len_empty = None
        while len_empty!=0:
            rows = self.send_parallel("dados",await_response=True).items()
            #Checking if any reactor didn't respond.
            empty = list(filter(lambda x: x[1] is None,rows))
            len_empty = len(empty)
            if len_empty!=0:
                #Error treatment in case some reactor fails to respond.
                empty = list(map(lambda x: x[0],empty))
                print(bcolors.FAIL+"[FAIL]","The following reactors didn't respond:"+"\n\t"+"\n\t".join(list(map(str,empty))))
                print("Resetting serial")
                sleep(10)
                for i in empty:
                    self.reactors[i].reset()
                    self.reactors[i].connect()
                print("Recovering last state")
                self.set_preset_state(path=PATHS.INITIAL_STATE_PATH)
                self.set_preset_state(path=CACHEPATH)
                sleep(10)
                print("Done"+bcolors.ENDC)

        rows = dict(map(lambda x: (x[0],OrderedDict(zip(self.header,x[1].split(" ")))),rows))
        if save_cache:
            self.log.cache_data(rows) #Index set to False because ID already exists in rows.
        return rows

    def log_dados(self,save_cache=True):
        """
        Logs output of `dados` in csv format.

        Args:
            save_cache (bool): Whether or not so save a cache file with the last reading with `log.log.cache_data`.
        """
        self.garbage()
        header = list(self.reactors.values())[0].send("cabecalho").split(" ")

        rows = self.send_parallel("dados",delay=13)
        rows = list(map(lambda x: (x[0],OrderedDict(zip(header,x[1].split(" ")))),rows))

        for _id,row in rows:
            self.log.log_rows(rows=[row],subdir=_id)
        rows = dict(rows)
        if save_cache:
            self.log.cache_data(rows) #Index set to False because ID already exists in rows.
        return rows
    
    def set_preset_state(self,path="preset_state.csv",sep="\t",chunksize=4, params=PATHS.REACTOR_PARAMETERS, **kwargs):
        """
        Prepare Arduinos with preset parameters from a csv file.
        Args:
            path (str): Path to the csv file.
            chunksize (int): How many to commands to send in a single line. A large value can cause Serial errors.
            sep (str): Column separator used in the csv.
        """
        df = pd.read_csv(path,sep=sep,index_col='ID',**kwargs)
        df = df[df.index.isin(self.ids)]
        df.columns = df.columns.str.lower() #Commands must be sent in lowercase
        #Dropping empty columns
        df.dropna(axis=1,inplace=True)
        if params:
            cols = list(set(df.columns)&set(params))
            df = df.loc[:,cols]
        #Setting schema
        schema = set(df.columns)&PATHS.SCHEMA.keys()
        schema = {k:PATHS.SCHEMA[k] for k in schema}
        df = df.astype(schema)
        with Pool(7) as p:
            p.map(set_in_chunks,map(lambda x: (self.reactors[x[0]],x[1],chunksize),df.to_dict(orient="index").items()))
        #Saving relevant parameters' values
        cols = list(set(df.columns)&set(PATHS.RELEVANT_PARAMETERS))
        self.preset_state = df.loc[:,cols]
        self.payload = self.preset_state.to_dict('index').copy()

    def calibrate(self,deltaT=120,dir="calibrate"):
        """
        Runs `curva` and dumps the result into txts.
        """
        if not os.path.exists(dir):
            os.mkdir(dir)
        out = {}
        self.send("curva",await_response=False)
        sleep(deltaT)
        for name,reactor in self.reactors.items():
            out[name] = reactor._conn.read_until('*** fim da curva dos LEDs ***'.encode('ascii'))
            with open(os.path.join(dir,f"reator_{name}.txt"),"w") as f:
                f.write(out[name].decode('ascii'))
        return out
    
    def __repr__(self):
        return f"{bcolors.OKCYAN}<Manager of reactors {''.join(str(self.ids))}>{bcolors.ENDC}"


if __name__ == '__main__':
    r = ReactorManager()