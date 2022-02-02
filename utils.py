import yaml

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