#Meant to be used in the raspberries
from crypt import methods
from urllib import response
from flask import Flask, request, jsonify
from serial import Serial
import serial.tools.list_ports as list_ports
from time import sleep
from pyduino.utils import bcolors
import logging
import re
import os
import socket
from typing import Any, Optional, Dict

logging.basicConfig(filename='slave.log', filemode='w', level=logging.DEBUG)

STEP = 1/8.0
HEADER_DELAY = 5

class ReactorServer(Flask):
    """
    Slave of HTTP Server to Serial handler.
    """

    def __init__(self, serial_port: Optional[str] = None, baudrate: int = 9600, *args: Any, **kwargs: Any):
        """
        Initializes the ReactorServer.

        Args:
            serial_port (str, optional): The serial port to connect to. If not specified, it searches for available devices.
            baudrate (int, optional): The baud rate for the serial connection.
            *args: Variable length arguments to pass to the Flask constructor.
            **kwargs: Keyword arguments to pass to the Flask constructor.
        """
        logging.debug("Creating server.")
        super().__init__(*args, **kwargs)
        self.connected = False
        self.reactor_id: Optional[int] = None
        self.port = serial_port
        self.baudrate = baudrate

        self.serial_connect()

        # Routes
        @self.route("/")
        def root():
            """
            Root route handler.
            """
            return "REACTOR SERVER", 200

        @self.route("/connect")
        def http_connect():
            """
            HTTP route for connecting to the reactor.
            """
            self.connect()
            return "OK", 200

        @self.route("/reset")
        def http_reset():
            """
            HTTP route for resetting the connection.
            """
            self.reset()
            return "OK", 200

        @self.route("/reboot")
        def reboot():
            """
            HTTP route for rebooting the server.
            """
            os.system("sudo reboot")
            return "OK", 200

        @self.route("/send", methods=['POST'])
        def http_send():
            """
            HTTP route for sending commands to the reactor.

            Returns:
                str: The response received from the reactor.
            """
            content = request.json
            logging.info(f"Received request: {content['command']}")
            if content['await_response']:
                response = self.send(content["command"], delay=content["delay"])
            else:
                response = self._send(content["command"])
            return jsonify({"response": response}), 200

        @self.route("/ping")
        def ping():
            """
            HTTP route for pinging the reactor.

            Returns:
                dict: A JSON object containing reactor information (id, serial_port, hostname).
            """
            digit_regex = r"(\d+)(?!.*\d)"
            hostname = socket.gethostname()
            digits = int(re.findall(digit_regex, hostname)[0])
            self.reactor_id = digits
            return jsonify({"id": self.reactor_id, "serial_port": self.port, "hostname": os.uname().nodename})

    def __delete__(self, _):
        self.serial.__del__()

    
    def serial_connect(self):
        if self.port is None:
            logging.debug("No serial port specified. Searching for available devices.")
            self.available_ports = list_ports.comports()
            self.available_ports = list(filter(lambda x: (x.vid,x.pid) in {(1027,24577),(9025,16),(6790,29987)},self.available_ports))
            self.port = self.available_ports[0].device
        self.serial = Serial(self.port, baudrate=self.baudrate, timeout=STEP)
        logging.info(f"Connected to serial port {self.serial}.")

    def connect(self, delay: float = STEP):
        """
        Begins the connection to the reactor.

        Args:
            delay (float, optional): Delay in seconds before sending the initial command.
        """
        sleep(HEADER_DELAY)
        self._recv(delay)
        self._send("quiet_connect")
        self.connected = True

    def reset(self):
        """
        Resets the connection to the reactor.
        """
        self.serial.flush()
        self.serial.close()
        self.serial.open()
        self.connected = True

    def close(self):
        """
        Interrupts the connection with the reactor.
        """
        if self.serial.is_open:
            self.send("fim")
            self.serial.close()

    def send(self, msg: str, delay: float = 0) -> str:
        """
        Sends a command to the reactor and receives the response.

        Args:
            msg (str): The command to send to the reactor.
            delay (float, optional): Delay in seconds before sending the command.

        Returns:
            str: The response received from the reactor.
        """
        if not self.connected:
            self.connect()
        self._send(msg)
        return self._recv()

    def _send(self, msg: str):
        """
        Sends a command to the reactor.

        Args:
            msg (str): The command to send to the reactor.
        """
        self.serial.write(msg.encode('ascii') + b'\n\r')

    def _recv(self) -> str:
        """
        Reads from the serial port until it finds an EOT ASCII token.

        Returns:
            str: The response received from the reactor.
        """
        response = self.serial.read_until(b'\x04') \
            .decode('ascii') \
            .strip("\n") \
            .strip("\r") \
            .strip("\x04")
        return response

    
    def __repr__(self):
        return f"{bcolors.OKCYAN}<Reactor at {self.port}>{bcolors.ENDC}"

    def set(self, data: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """
        Sets the value of variables based on a dictionary.

        Args:
            data (dict, optional): Dictionary containing variable-value pairs.
            **kwargs: Additional variable-value pairs.

        Examples:
            >>> reator.set({"440": 50, "brilho": 100})
        """
        data = {**(data or {}), **kwargs}
        args = ",".join(f'{k},{v}' for k, v in data.items())
        cmd = f"set({args})"
        self._send(cmd)

    def get(self, key: Optional[str] = None) -> Any:
        """
        Returns the value of a variable or variables.

        Args:
            key (str, optional): The key of the variable to retrieve. If not specified, returns all variables.

        Returns:
            Any: The value of the variable(s).
        """
        if key is None:
            return self._get_all()
        if isinstance(key, str):
            key = [key]
        return

    def _get_all(self) -> Any:
        """
        Returns the values of all variables.

        Returns:
            Any: The values of all variables.
        """
        resp = self.send("get")
        # Parse the response and return the values of all variables.
        return resp

if __name__=="__main__":
    rs = ReactorServer(import_name="Pyduino Slave Server")
    rs.run(port=5000,host="0.0.0.0")