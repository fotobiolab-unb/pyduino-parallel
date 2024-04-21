from pyduino.paths import PATHS
from pyduino.utils import get_servers
from pyduino.spectra import Spectra, SpectraManager
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="Path to config YAML file. If none, uses default path.")

args = parser.parse_args()

if args.config:
    SPECTRUM_PATH = os.path.join(args.config)
    PATHS.read(SPECTRUM_PATH)

if __name__=="__main__":
    if PATHS.SYSTEM_PARAMETERS.get("partition", "all") == "all":
        spectra = Spectra(**PATHS.HYPERPARAMETERS)
        g = Spectra(**PATHS.HYPERPARAMETERS)
    elif PATHS.SYSTEM_PARAMETERS.get("partition", "all") == "single":
        managers = {}
        servers = get_servers()
        for id, host in servers.items():
            hyper = PATHS.HYPERPARAMETERS
            hyper.update({"include": {id:host}})
            p = Spectra(**hyper)
            managers[id] = p

        g = SpectraManager(managers)