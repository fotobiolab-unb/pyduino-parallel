import serial
from operator import attrgetter
from time import sleep, time
import pandas as pd
import serial.tools.list_ports
import re
from collections import OrderedDict
from log import log

STEP = 1 / 16
COLN = 48 #Number of columns to parse from Arduino (used for sanity tests)

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

    def close(self):
        """
        Interrompe conexão.
        """
        if self._conn.is_open:
            self.send("fim")
            self._conn.close()

    def send(self, msg, delay=STEP):
        """
        Envia mensagem para o reator e retorna resposta.
        """
        if not self.connected:
            self.connect()
        self._send(msg)
        return self._recv(delay)

    def _send(self, msg):
        self._conn.write(msg.encode('ascii') + b'\n\r')
        # self._send_cb(msg)

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
        args = ', '.join(f'{k}, {v}' for k, v in data.items())
        self.send(f'set({args})')

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

class ReactorManager:
    pinged = False
    def __init__(self,baudrate=9600):
        self.available_ports = serial.tools.list_ports.comports()
        self.ports = list(filter(lambda x: (x.vid,x.pid) in {(1027,24577),(9025,16),(6790,29987)},self.available_ports))
        self.reactors = {x.device:Reator(port=x.device,baudrate=baudrate,cb=(lambda inpt: print(f'>>> {inpt}'), lambda out: print(out, end='\n[END]\n'))) for x in self.ports}
        self._id = {}

        #Init
        self.connect()
        self.garbage()
        self.ping()
        self.log_init()
        #Info
        print("\n".join(map(lambda x: f"Reactor {x[0]} at port {x[1]}",self._id.items())))
    
    def send(self,command,await_response=True,**kwargs):
        out = {}
        for k,r in self.reactors.items():
            #print(k)
            if await_response:
                out[k] = r.send(command,**kwargs)
            else:
                r._send(command)
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
        self.pinged = True

    def connect(self):
        for k,r in self.reactors.items():
            r.connect()
    
    def start(self):
        self.connect()
        self.ping()

    #Logging

    def log_init(self):
        """
        Creates log directories for each Arduino.
        """
        if not self.pinged:
            self.ping()

        self.log = log(subdir=list(self._id.keys()))
    
    def log_dados(self):
        """
        Logs output of `dados` in csv format.
        """
        self.garbage()
        for port,reactor in self.reactors.items():
            n = 0
            r_len, h_len = float("nan"),float("nan")
            while r_len != COLN or h_len != COLN:
                #print(f"[INFO]({n}) Reading reactor {self._id_reverse[port]} on port {port}")
                response = reactor.send("dados",delay=0.5).split(" ")
                header = reactor.send("cabecalho",delay=0.5).split(" ")
                r_len, h_len = len(response),len(header)
                n+=1
                #print("[INFO]","[SIZE]",len(response),len(header))
            row = OrderedDict(zip(header,response))
            self.log.log_rows(rows=[row],subdir=self._id_reverse[port],sep='\t',index=False)

if __name__ == '__main__':
    r = ReactorManager()