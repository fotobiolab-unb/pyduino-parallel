from pyduino.paths import PATHS
import yaml
from nmap import PortScanner
import requests
import numpy as np
from collections import OrderedDict
from urllib.parse import urljoin
import logging

logging.basicConfig(filename='pyduino.log', filemode='w', level=logging.DEBUG)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_param(data, key: str, ids: set = False) -> OrderedDict:

    """
    Retrieve a specific parameter from a dictionary of data.

    Parameters:
    - data: The dictionary containing the data.
    - key: The key of the parameter to retrieve.
    - ids: (optional) A set of IDs to filter the data. If not provided, all data will be returned.

    Returns:
    - An ordered dictionary containing the filtered data.

    """
    filtered = OrderedDict(list(map(lambda x: (x[0], x[1][key]), data.items())))
    if not ids:
        return filtered
    else:
        return OrderedDict(filter(lambda x: x[0] in ids,filtered.items()))

def yaml_get(filename):
    """
    Loads hyperparameters from a YAML file.
    """
    y = None
    with open(filename) as f:
        y = yaml.load(f.read(),yaml.Loader)
    return y


def ReLUP(x):
    """Computes probability from an array X after passing it through a ReLU unit (negatives are zero).

    Args:
        x (numpy.array): Input Array
    """
    x_relu = x.copy()
    x_relu[x_relu<0] = 0

    if x_relu.sum() == 0:
        return np.ones_like(x_relu)/len(x_relu)
    else:
        return x_relu/x_relu.sum()

def get_meta(url):
    resp = requests.get(urljoin(url, "ping"))
    if resp.ok :
        return resp.json()
    else:
        logging.error(f"Unable to connect to {url}")
        raise ConnectionRefusedError(url)

def get_servers(
        net=PATHS.SYSTEM_PARAMETERS.get("network", "192.168.1.1/24"),
        port=PATHS.SYSTEM_PARAMETERS.get("port", 5000),
        exclude=PATHS.SYSTEM_PARAMETERS.get("exclude", None)
    )->dict:
    """
    Get a dictionary of available servers in the network.

    Args:
        net (str): The network address range to scan for servers. Default is "192.168.0.1/24".
        port (str): The port number to scan for servers. Default is "5000".
        exclude (str): IP addresses to exclude from the scan. Default is None.

    Returns:
        dict: A dictionary of available servers, where the keys are the server IDs and the values are the server URLs.

    """
    logging.debug(f"Searching for devices on {net}:{port}")
    port_scanner = PortScanner()
    args = "--open" if exclude is None else f"--open --exclude {exclude}"
    results = port_scanner.scan(net, str(port), arguments=args, timeout=60)
    hosts = list(map(lambda x: f"http://{x}:{str(port)}", results["scan"].keys()))
    servers = {}
    for host in hosts:
        try:
            v = requests.get(host, timeout=2).text == "REACTOR SERVER"
            meta = get_meta(host)
            if v:
                if meta["id"] in servers:
                    logging.warning(f"Duplicate ID found: {meta['id']}")
                servers[meta["id"]] = host
        except:
            pass
    logging.debug(f"Found {len(servers)} devices")
    return servers

class TriangleWave:
    def __init__(self,p_0: float, p_i: float, p_f: float, N: int):
        """Generates a triangular wave according to the formula:

        Q\left(x\right)=(N-\operatorname{abs}(\operatorname{mod}\left(x,2N\right)-N))\left(\frac{p_{f}-p_{i}}{N}\right)+p_{i}

        Args:
            p_0 (float): Initial value at n=0
            p_i (float): Lower bound
            p_f (float): Upper bound
            N (int): Steps to reach upper bound
        """
        self.N = N
        self.p_0 = p_0
        self.p_i = p_i
        self.p_f = p_f

        self.a = N*(self.p_0 - self.p_i)/(self.p_f - self.p_i)#Phase factor
    
    def Q(self,x: int):
        return (self.N - abs((x%(2*self.N))-self.N))*(self.p_f - self.p_i)/self.N + self.p_i
    
    def y(self,x: int):
        return self.Q(x + self.a)

def partition(X: np.ndarray, n: int) -> list:
    """
    Partitions the array `X` in blocks of size `n` except the last.

    Args:
        X (numpy.array): Input 2D array
        n (int): Number of partitions
    
    Returns:
        list: A list containing the array partitions.
    """
    assert X.ndim == 2, "X must be a matrix"
    #Number of partitions
    r = X.shape[0] % n
    m = X.shape[0] // n + (r > 0)
    X_enlarged = np.pad(X, ((0, n*m - X.shape[0]), (0,0)), constant_values=0)
    X_split = np.array_split(X_enlarged, m)
    if r > 0:
        X_split[-1] = X_split[-1][:r,:]
    return X_split
