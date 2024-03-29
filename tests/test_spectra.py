import sys
import os
import numpy as np
import pandas as pd
import dotenv
dotenv.load_dotenv()
dotenv.load_dotenv(dotenv_path=".env.local")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyduino.spectra import Spectra, PATHS

class TestSpectra:
    g = Spectra(**PATHS.HYPERPARAMETERS)
    g.deltaTgotod = 1
    g.deltaT = 1
    g.init()
    
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
    def test_logger(self):
        self.g.GET("test")    
    def test_log_tensor(self):
        # Create a mock writer object
        class MockWriter:
            def __init__(self):
                self.scalar_values = {}

            def add_scalar(self, name, value, step):
                if name not in self.scalar_values:
                    self.scalar_values[name] = {}
                self.scalar_values[name][step] = value

        # Create an instance of the Spectra class
        g = Spectra(**PATHS.HYPERPARAMETERS)
        g.init()
        g.y = list(range(len(g.parameters)))

        # Create a mock writer object and assign it to the Spectra instance
        mock_writer = MockWriter()
        g.writer = mock_writer

        # Call the log_tensor method
        g.log_tensor(0)

        # Assert that the scalar values were logged correctly
        for j, params in enumerate(g.view_g()):
            for param_name, param in zip(g.parameters, params):
                expected_name = f"{param_name}/{j}"
                expected_value = param
                assert mock_writer.scalar_values[expected_name][0] == expected_value

            expected_fitness_name = f"Fitness/{j}"
            expected_fitness_value = g.y[j]
            assert mock_writer.scalar_values[expected_fitness_name][0] == expected_fitness_value