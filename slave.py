#Meant to be used in the raspberries
from crypt import methods
from urllib import response
from flask import Flask, request, jsonify
from serial import Serial
import serial.tools.list_ports as list_ports
from time import sleep
from utils import bcolors
import logging
import re
import os

logging.basicConfig(filename='slave.log', filemode='w', level=logging.DEBUG)

STEP = 1/8.0
HEADER_DELAY = 5

class ReactorServer(Flask):
    """
    Slave of HTTP Server to Serial handler.
    """

    def __init__(self, serial_port = None, baudrate=9600,*args,**kwargs):
        logging.debug("Creating server.")
        super().__init__(*args,**kwargs)
        self.connected = False
        self.reactor_id = None
        self.port = serial_port
        if serial_port is None:
            logging.debug("No serial port specified. Searching for available devices.")
            self.available_ports = list_ports.comports()
            self.available_ports = list(filter(lambda x: (x.vid,x.pid) in {(1027,24577),(9025,16),(6790,29987)},self.available_ports))
            self.port = self.available_ports[0].device
        self.serial = Serial(self.port, baudrate=baudrate, timeout=STEP)
        logging.info(f"Connected to serial port {self.serial}.")

        #Routes
        @self.route("/")
        def root():
            return "REACTOR SERVER", 200

        @self.route("/connect")
        def http_connect():
            self.connect()
            return "OK", 200
        
        @self.route("/reset")
        def http_reset():
            self.reset()
            return "OK", 200

        @self.route("/send",methods=['POST'])
        def http_send():
            content = request.json
            logging.info(f"Received request: {content['command']}")
            if content['await_response']:
                response = self.send(content["command"],delay=content["delay"])
            else:
                response = self._send(content["command"])
            return jsonify({"response":response}),200
        
        @self.route("/ping")
        def ping():
            if not self.reactor_id:
                if not self.connected:
                    self.connect()
                response = self.send("ping",delay=1)
                self.reactor_id = int(re.findall(r"\d+?",response)[0])
            return jsonify({"id":self.reactor_id,"serial_port":self.port,"hostname":os.uname().nodename})         

    def __delete__(self, _):
        self.serial.__del__()

    def connect(self,delay=STEP):
        """
        Connection begin.
        """
        sleep(HEADER_DELAY)
        self._recv(delay)
        self._send("quiet_connect")
        self.connected = True

    def reset(self):
        """
        Resets connection.
        """
        self.serial.flush()
        self.serial.close()
        self.serial.open()
        self.connected = True

    def close(self):
        """
        Interrompe conexão.
        """
        if self.serial.is_open:
            self.send("fim")
            self.serial.close()

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
        self.serial.write(msg.encode('ascii') + b'\n\r')

    def _recv(self,delay=STEP):
        out = []
        for _ in range(256):
            sleep(delay)
            new = self.serial.read_until()
            if new:
                new = new.decode('ascii').strip("\n").strip("\r")
                out.append(new)
            if out and not new:
                resp = ''.join(out)
                return resp
    
    def __repr__(self):
        return f"{bcolors.OKCYAN}<Reactor at {self.port}>{bcolors.ENDC}"

    def set(self, data=None, **kwargs):
        """
        Define o valor de todas as variáveis a partir de um dicionário.

        Exemplo:
            >>> reator.set({"440": 50, "brilho": 100})
        """
        data = {**(data or {}), **kwargs}
        args = ",".join(f'{k},{v}' for k, v in data.items())
        cmd = f"set({args})"
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

if __name__=="__main__":
    rs = ReactorServer(import_name="a")
    rs.run(port=5000,host="0.0.0.0")