from pyduino.paths import PATHS
from pyduino.utils import get_servers
from pyduino.spectra import Spectra, SpectraManager
from datetime import datetime
from tensorboardX import SummaryWriter
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
        g.log.backup_config_file()
    elif PATHS.SYSTEM_PARAMETERS.get("partition", "all") == "single":
        tensorboard_path = os.path.join(
            "./log",
            datetime.today().strftime("%Y"),
            datetime.today().strftime("%m"),
            PATHS.HYPERPARAMETERS["log_name"],
            "runs",
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        writer = SummaryWriter(tensorboard_path)
        managers = {}
        servers = get_servers()
        backed_up_config_file = False
        for id, host in servers.items():
            hyper = PATHS.HYPERPARAMETERS
            hyper.update({"include": {id:host}})
            hyper.update({"summary_writer": writer})
            p = Spectra(**hyper)
            if not backed_up_config_file:
                p.log.backup_config_file()
                backed_up_config_file = True
            managers[id] = p

        g = SpectraManager(managers)