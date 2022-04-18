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