import os
import yaml
import warnings

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def yaml_get(filename):
    """
    Loads hyperparameters from a YAML file.
    """
    y = None
    with open(filename) as f:
        y = yaml.load(f.read(),yaml.Loader)
    return y

class Paths():
    def read(self,config_path):
        self.CONFIG_PATH = config_path
        self.SYSTEM_PARAMETERS = yaml_get(self.CONFIG_PATH)['system']
        self.SLAVE_PARAMETERS = yaml_get(self.CONFIG_PATH)['slave']
        self.RELEVANT_PARAMETERS = self.SYSTEM_PARAMETERS['relevant_parameters']
        self.HYPERPARAMETERS = yaml_get(self.CONFIG_PATH)['hyperparameters']
        self.INITIAL_STATE_PATH = os.path.join(__location__,self.SYSTEM_PARAMETERS['initial_state'])
        self.SCHEMA = self.SYSTEM_PARAMETERS["standard_parameters"]
        self.REACTOR_PARAMETERS = list(self.SCHEMA.keys())
        self.TENSORBOARD = self.SYSTEM_PARAMETERS.get("tensorboard", None)

PATHS = Paths()
if os.path.exists(os.path.join(__location__,"config.yaml")):
    PATHS.read(os.path.join(__location__,"config.yaml"))
else:
    warnings.warn("No config.yaml file found in the current directory. Please create one.")
    PATHS.read(os.path.join(__location__,"config.template.yaml"))