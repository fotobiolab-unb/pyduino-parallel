import yaml
from nmap import PortScanner
import requests
import numpy as np
from collections import OrderedDict

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

def get_servers(net="192.168.0.1/24",port="5000",exclude=None):
    port_scanner = PortScanner()
    args = "--open" if exclude is None else f"--open --exclude {exclude}"
    results = port_scanner.scan(net,port,arguments=args,timeout=60)
    hosts = list(map(lambda x: f"http://{x}:{str(port)}",results["scan"].keys()))
    servers = []
    for host in hosts:
        try:
            v = requests.get(host,timeout=2).text == "REACTOR SERVER"
            if v:
                servers.append(host)
        except:
            pass
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