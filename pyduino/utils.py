import yaml
from nmap import PortScanner
import requests

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

def yaml_get(filename):
    """
    Loads hyperparameters from a YAML file.
    """
    y = None
    with open(filename) as f:
        y = yaml.load(f.read(),yaml.Loader)
    return y

def get_servers(net="192.168.0.1/24",port="5000"):
    port_scanner = PortScanner()
    results = port_scanner.scan(net,port,"--open",timeout=60)
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