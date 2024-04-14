import sys
import os
import numpy as np
import dotenv
import dotenv
import pytest
dotenv.load_dotenv()
dotenv.load_dotenv(dotenv_path=".env.local")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyduino.optimization.nelder_mead import NelderMead

class TestOptim:
    optimizer = NelderMead(population_size=10, ranges=[(-10, 10)] * 4)

    def test_logic(self):
        x = (2*np.random.random((10, 4))-1)*100

        print("X")
        print(x)
        print("Forward -> Backward")
        print(self.optimizer.backward(self.optimizer.forward(x)))

        assert np.allclose(x, self.optimizer.backward(self.optimizer.forward(x)))
        assert np.allclose(x, self.optimizer.inverse_view(self.optimizer.view(x)))

        assert np.all(self.optimizer.backward(np.zeros((10,4))) < 1)

    def test_nelder_mead_parabola(self):
        class Oracle:
            def ask(self, X):
                return (X**2).sum(axis=1)
        oracle = Oracle()
        self.optimizer.ask_oracle = oracle.ask

        # Set initial population to oracle
        self.optimizer.init_oracle()

        for i in range(100):
            self.optimizer.step()
        assert self.optimizer.view(self.optimizer.population).shape == (10, 4)
        assert self.optimizer.view_g().shape == (10, 4)
        assert self.optimizer.inverse_view(self.optimizer.view(self.optimizer.population)).shape == (10, 4)
        assert np.isclose(self.optimizer.y.min(), 0, atol=1e-4), f"Oracle: {self.optimizer.y.min()}"

    def test_nelder_mead_rosenbrock(self):
        class Oracle:
            def ask(self, X):
                return ((1 - X[:, :-1])**2).sum(axis=1) + 100 * ((X[:, 1:] - X[:, :-1]**2)**2).sum(axis=1)
        oracle = Oracle()
        self.optimizer.ask_oracle = oracle.ask

        # Set initial population to oracle
        self.optimizer.init_oracle()

        for i in range(1000):
            self.optimizer.step()
        assert self.optimizer.view(self.optimizer.population).shape == (10, 4)
        assert self.optimizer.view_g().shape == (10, 4)
        assert self.optimizer.inverse_view(self.optimizer.view(self.optimizer.population)).shape == (10, 4)
        assert np.isclose(self.optimizer.y.min(), 0, atol=1e-4), f"Oracle: {self.optimizer.y.min()}"