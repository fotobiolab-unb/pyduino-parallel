import sys
import os
import numpy as np

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
    def test_update_fitness(self):
        self.g.update_fitness()
    def test_oracle(self):
        data = self.g.F_get()
        for k in data.keys():
            data[k][self.g.fparam] = '0'
        df = self.g.pretty_print_dict(data)
        y = self.g.ask_oracle(self.g.population)
    def test_logger(self):
        self.g.GET("test")