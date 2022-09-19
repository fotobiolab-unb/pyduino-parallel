from pyduino.spectra import Spectra, hyperparameters, PATHS
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="Path to config YAML file. If none, uses default path.")

args = parser.parse_args()

if args.config:
    SPECTRUM_PATH = os.path.join(args.config)
    PATHS.read(SPECTRUM_PATH)

if __name__=="__main__":
    g = Spectra(**hyperparameters)