import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyduino.optimization.nelder_mead import NelderMead

def test_nelder_mead_parabola():
    optimizer = NelderMead(population_size=10, ranges=[(-10, 10)] * 4)
    class Oracle:
        def set(self, X):
            self.X = X
        def ask(self):
            return (self.X**2).sum(axis=1)
    oracle = Oracle()
    optimizer.ask_oracle = oracle.ask
    optimizer.set_oracle = lambda X: oracle.set(X)

    # Set initial population to oracle
    optimizer.init_oracle()

    for i in range(100):
        optimizer.step()
    assert optimizer.view(optimizer.population).shape == (10, 4)
    assert optimizer.view_g().shape == (10, 4)
    assert optimizer.inverse_view(optimizer.view(optimizer.population)).shape == (10, 4)
    assert np.isclose(optimizer.ask_oracle().min(), 0, atol=1e-4), f"Oracle: {optimizer.ask_oracle().min()}"

def test_nelder_mead_rosenbrock():
    optimizer = NelderMead(population_size=10, ranges=[(-10, 10)] * 4)
    class Oracle:
        def set(self, X):
            self.X = X
        def ask(self):
            return ((1 - self.X[:, :-1])**2).sum(axis=1) + 100 * ((self.X[:, 1:] - self.X[:, :-1]**2)**2).sum(axis=1)
    oracle = Oracle()
    optimizer.ask_oracle = oracle.ask
    optimizer.set_oracle = lambda X: oracle.set(X)

    # Set initial population to oracle
    optimizer.init_oracle()

    for i in range(1000):
        optimizer.step()
    assert optimizer.view(optimizer.population).shape == (10, 4)
    assert optimizer.view_g().shape == (10, 4)
    assert optimizer.inverse_view(optimizer.view(optimizer.population)).shape == (10, 4)
    assert np.isclose(optimizer.ask_oracle().min(), 0, atol=1e-4), f"Oracle: {optimizer.ask_oracle().min()}"