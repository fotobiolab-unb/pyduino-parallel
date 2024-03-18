import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyduino.spectra import Spectra, PATHS

class TestSpectra:
    g = Spectra(**PATHS.HYPERPARAMETERS)
    g.deltaTgotod = 1
    g.deltaT = 1
    def test_functions(self):
        assert len(self.g.reactors) != 0
        assert len(self.g.ids) == self.g.population_size

    def test_array_assignment(self):
        X = np.random.random((len(self.g.ids),len(self.g.parameters)))
        assigned = self.g.assign_to_reactors(X)
        keys = list(assigned.keys())
        assert len(keys)==len(self.g.ids)
        assert len(assigned[keys[0]]) == len(self.g.parameters)
    def test_oracle(self):
        data = self.g.F_get()
        df = self.g.pretty_print_dict(data)
        assert isinstance(df, pd.DataFrame)
        y = self.g.ask_oracle(self.g.population)
        self.g.update_fitness(y)
    def test_logger(self):
        self.g.GET("test")