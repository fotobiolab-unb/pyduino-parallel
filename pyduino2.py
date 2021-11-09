import serial
from operator import attrgetter
from time import sleep
import pandas as pd
import serial.tools.list_ports

STEP = 1 / 32


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
        self._conn = serial.Serial(port, baudrate=baudrate)
        
        self._send_cb, self._recv_cb = cb or (lambda inpt, out: None) 
        self._send_cb = self._send_cb or (lambda _: None)
        self._recv_cb = self._recv_cb or (lambda _: None)

    def __delete__(self, _):
        self.close()


    def connect(self):
        """
        Inicia conexão.
        """
        self._recv()
        self._send("manual_connect")
        self._recv()
        self.connected = True

    def close(self):
        """
        Interrompe conexão.
        """
        if self._conn.is_open:
            self.send("fim")
            self._conn.close()

    def send(self, msg):
        """
        Envia mensagem para o reator e retorna resposta.
        """
        if not self.connected:
            self.connect()
        self._send(msg)
        return self._recv()

    def _send(self, msg):
        self._conn.write(msg.encode('ascii') + b'\n\r')
        self._send_cb(msg)

    def _recv(self):
        out = []
        for _ in range(256):
            sleep(STEP)
            new = self._conn.read_all()
            if new:
                new = new.decode('ascii').strip()
                out.append(new)
            # print('new:', new, out)
            if out and not new:
                resp = ''.join(out)
                self._recv_cb(resp)
                return resp

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
    def __init__(self,baudrate=9600):
        self.available_ports = serial.tools.list_ports.comports()
        self.reactors = {}
        for port in self.available_ports:
            try:
                r = Reator(port=port.device,baudrate=baudrate,cb=(lambda inpt: print(f'>>> {inpt}'), lambda out: print(out, end='\n\n')))
                r.connect()
                r._recv()
                ping = r.send("ping")
                print(ping)
                r_id = int(r.send("ping\n").split('\n')[0].strip())
                self.reactors[r_id] = r 
            except serial.SerialException as e:
                print(e)
    
    def send(self,command):
        out = {}
        for k,r in self.reactors.items():
            out[k] = r.send(command)
        return out
    def set(self, data=None, **kwargs):
        for k,r in self.reactors.items():
            r.set(data=None, **kwargs)
    def get(self,key=None):
        for k,r in self.reactors.items():
            r.get(key=None)


if __name__ == '__main__':
    df = pd.read_csv("COM_order.txt",sep="\t",header=None)
    N = df.iloc[:,1].tolist()
    df = df.set_index(0)
    df = [0] + df.iloc[:,0].tolist()
    ports = [f"COM{n}" for n in N]
    N = [44,39,41,42,43,38,40]
    #reactors = {n:Reator(f"COM{n}",cb=(lambda inpt: print(f'>>> {inpt}'), lambda out: print(out, end='\n\n'))) for n in N}

    def func(command):
        for i,r in enumerate(reactors.values()):
            r.send(command+"\n")

    def dual_func_delay(command1,command2,delay):
        for r in reactors.values():
             r.send(command1)
             sleep(delay)
             r.send(command2)
    
    def xmas(params,delay,interval):
        for c in params:
            for b in interval:
                func(f"set({c},{b})")
                sleep(delay)
        func(f"set({c},0)")