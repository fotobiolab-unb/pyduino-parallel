import serial
from operator import attrgetter
from time import sleep, time
import pandas as pd
import serial.tools.list_ports
import re
from collections import OrderedDict
from log import log
import pandas as pd
from multiprocessing import Pool, Process
from functools import partial
import os
from bcolors import bcolors

STEP = 1 / 16
COLN = 48 #Number of columns to parse from Arduino (used for sanity tests)
CACHEFILE = "cache.csv"
REACTOR_PARAMETERS = None
if os.path.exists("parameters.txt"):
    with open("parameters.txt") as file:
        REACTOR_PARAMETERS = set(map(lambda x : x.strip(),file.readlines()))

#From https://stackoverflow.com/a/312464/6451772
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class param(property):
    def __init__(self, typ=int, name=None):
        self.name = name
        self.type = typ
        param = self
            
        def fget(self):
            return param.type(self.get(param.name))

        def fset(self, value):
            self.set({param.name: value})

        super().__init__(fget, fset)

    def __set_name__(self, cls, name):
        if self.name is None:
            self.name = name


class Reator:
    """
    Controla o reator via protocolo LVP. 
    """

    port = property(attrgetter('_conn.port'))
    baudrate = property(attrgetter('_conn.baudrate'))

    temp = param(float)
    brilho = param()
    ar = param()
    ima = param()
    color_branco = param(name="branco")
    color_440 = param(name="440")

    def __init__(self, port, baudrate=9600, cb=None):
        self.connected = False
        self._conn = serial.Serial(port, baudrate=baudrate, timeout=STEP)
        self._port = port

    def __delete__(self, _):
        self.close()


    def connect(self,delay=STEP):
        """
        Inicia conexão.
        """
        self._recv(delay)
        self._send("quiet_connect")
        #self._recv(delay)
        self.connected = True

    def reset(self):
        """
        Resets connection.
        """
        self._conn.close()
        self._conn.open()
        self.connected = True

    def close(self):
        """
        Interrompe conexão.
        """
        if self._conn.is_open:
            self.send("fim")
            self._conn.close()

    def send(self, msg, delay=0, recv_delay=STEP):
        """
        Envia mensagem para o reator e retorna resposta.

        Args:
            delay (int): Delay in seconds between sending and reading.
            recv_delay (int): Delay in seconds sent to recv.
        """
        if not self.connected:
            self.connect()
        self._send(msg)
        sleep(delay)
        return self._recv(recv_delay)

    def _send(self, msg):
        self._conn.write(msg.encode('ascii') + b'\n\r')
        # self._send_cb(msg)
    
    def set_in_chunks(self,params,chunksize=4):
        """
        Sets params into chunks in a different thread.
        """
        p = Process(target=set_chunk,args=(self,params,2,chunksize))
        p.start()

    def _recv(self,delay=STEP):
        out = []
        for _ in range(256):
            sleep(delay)
            #new = self._conn.read_all()
            new = self._conn.read_until()
            if new:
                new = new.decode('ascii').strip("\n").strip("\r")
                out.append(new)
            # print('new:', new, out)
            if out and not new:
                resp = ''.join(out)
                # self._recv_cb(resp)
                return resp
    
    def __repr__(self):
        return f"<Reactor at {self._port}>"

    def set(self, data=None, **kwargs):
        """
        Define o valor de todas as variáveis a partir de um dicionário.

        Exemplo:
            >>> reator.set({"440": 50, "brilho": 100})
        """
        data = {**(data or {}), **kwargs}
        args = ",".join(f'{k},{v}' for k, v in data.items())
        cmd = f"set({args})"
        #print("[INFO]","SEND",cmd)
        self._send(cmd)

    def get(self, key=None):
        """
        Retorna o valor de uma ou mais variáveis.
        """
        if key is None:
            return self._get_all()
        if isinstance(key, str):
            key = [key]
        return

    def _get_all(self):
        resp = self.send("get")

def send_wrapper(reactor_item,command,delay,await_response=True):
    if await_response:
        return (reactor_item[0],reactor_item[1].send(command,delay=delay))
    else:
        reactor_item[0],reactor_item[1]._send(command)
        sleep(delay)
        return True

def set_chunk(reactor,params,delay,chunksize):
        """
        Sets params into chunks in a different thread.
        """
        params = list(params)
        for chunk in chunks(params,chunksize):
            cmd = ",".join(list(map(lambda u: f"{u[0]},{u[1]}",chunk)))
            cmd = f"set({cmd})"
            reactor._send(cmd)
            sleep(delay)

class ReactorManager:
    pinged = False
    def __init__(self,baudrate=9600,log_name=None):
        self.available_ports = serial.tools.list_ports.comports()
        self.ports = list(filter(lambda x: (x.vid,x.pid) in {(1027,24577),(9025,16),(6790,29987)},self.available_ports))
        self.reactors = {x.device:Reator(port=x.device,baudrate=baudrate,cb=(lambda inpt: print(f'>>> {inpt}'), lambda out: print(out, end='\n[END]\n'))) for x in self.ports}
        self._id = {}

        #Init
        self.connect()
        self.garbage()
        self.ping()
        #Info
        print("\n".join(map(lambda x: f"Reactor {x[0]} at port {x[1]}",self._id.items())))
        self.header = None
    
    def send(self,command,await_response=True,**kwargs):
        out = {}
        for k,r in self.reactors.items():
            if await_response:
                out[k] = r.send(command,**kwargs)
            else:
                r._send(command)
        return out
    
    def send_parallel(self,command,delay,await_response=True):
        out = []
        with Pool(7) as p:
            out = p.map(partial(send_wrapper,command=command,delay=delay,await_response=await_response),list(self.reactors.items()))
        return out

    def garbage(self):
        """
        Reads data from Arduino and discards it.
        """
        for k,r in self.reactors.items():
            r._conn.reset_output_buffer()

    def set(self, data=None, **kwargs):
        for k,r in self.reactors.items():
            r.set(data=data, **kwargs)
    def get(self,key=None):
        for k,r in self.reactors.items():
            r.get(key=key)
    
    def ping(self):
        responses = self.send("ping",delay=1)
        for name,res in responses.items():
            if isinstance(res,str):
                i = int(re.findall(r"\d+?",res)[0])
                self._id[i] = name
        self._id_reverse = {v:k for k,v in self._id.items()}
        self._id = OrderedDict(self._id)
        self.pinged = True

    def connect(self):
        for k,r in self.reactors.items():
            r.connect()
    
    def start(self):
        self.connect()
        self.ping()

    #Logging

    def log_init(self,**kwargs):
        """
        Creates log directories for each Arduino.
        """
        if not self.pinged:
            self.ping()

        self.log = log(subdir=list(self._id.keys()),**kwargs)
    
    def dados(self,save_cache=True):
        """
        Get data from Arduinos.

        Args:
            save_cache (bool): Whether or not so save a cache file with the last reading with `log.log.cache_data`.
        """
        self.garbage()

        if self.header is None:
            self.header = list(self.reactors.values())[0].send("cabecalho").split(" ")

        len_empty = None
        while len_empty!=0:
            rows = self.send_parallel("dados",delay=20)
            #Checking if any reactor didn't respond.
            empty = list(filter(lambda x: x[1] is None,rows))
            len_empty = len(empty)
            if len_empty!=0:
                #Error treatment in case some reactor fails to respond.
                empty = list(map(lambda x: x[0],empty))
                print(bcolors.FAIL+"[FAIL]","The following reactors didn't respond:"+"\n"+"\n\t".join(empty))
                print("Resetting serial")
                for port in empty:
                    self.reactors[port].reset()
                sleep(2)
                print("Recovering last state")
                self.set_preset_state(path=CACHEFILE)
                print("Done")

        rows = dict(map(lambda x: (self._id_reverse[x[0]],OrderedDict(zip(self.header,x[1].split(" ")))),rows))
        if save_cache:
            self.log.cache_data(rows,sep='\t',index=False) #Index set to False because ID already exists in rows.
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
        rows = list(map(lambda x: (self._id_reverse[x[0]],OrderedDict(zip(header,x[1].split(" ")))),rows))

        for _id,row in rows:
            self.log.log_rows(rows=[row],subdir=_id,sep='\t',index=False)
        rows = dict(rows)
        if save_cache:
            self.log.cache_data(rows,sep='\t',index=False) #Index set to False because ID already exists in rows.
        return rows
    
    def set_preset_state(self,path="preset_state.csv",sep="\t",chunksize=4,**kwargs):
        """
        Prepare Arduinos with preset parameters from a csv file.
        Args:
            path (str): Path to the csv file.
            chunksize (int): How many to commands to send in a single line. A large value can cause Serial errors.
            sep (str): Column separator used in the csv.
        """
        df = pd.read_csv(path,sep=sep,index_col='ID',**kwargs)
        df.columns = df.columns.str.lower() #Commands must be sent in lowercase
        if REACTOR_PARAMETERS:
            cols = list(set(df.columns)&(REACTOR_PARAMETERS))
            df = df.loc[:,cols]
        for i,row in df.iterrows():
            row = list(row[~row.isna()].astype(int).items())
            i = int(i)
            self.reactors[self._id[i]].set_in_chunks(row,chunksize)
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
            with open(os.path.join(dir,f"reator_{self._id_reverse[name]}.txt"),"w") as f:
                f.write(out[name].decode('ascii'))
        return out

if __name__ == '__main__':
    r = ReactorManager()
    r.set_preset_state()