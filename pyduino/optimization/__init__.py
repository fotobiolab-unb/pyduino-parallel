from typing import Callable
from abc import ABC, abstractmethod
import warnings
import numpy as np

def linmap(domain: list[tuple[float, float]], codomain: list[tuple[float, float]]) -> Callable[[np.ndarray], np.ndarray]:
    """
    Linear mapping from domain to codomain.
    domain list of pairs:
        Maxima and minima that each parameter can cover.
        Example: [(0, 10), (0, 10)] for two parameters ranging from 0 to 10.
    codomain list of pairs:
        Maxima and minima that each parameter can cover.
        Example: [(0, 10), (0, 10)] for two parameters ranging from 0 to 10.
    """
    domain = np.array(domain)
    codomain = np.array(codomain)

    M = (codomain[:, 1] - codomain[:, 0])/ (domain[:, 1] - domain[:, 0])

    def f(x):
        assert x.shape[1] == domain.shape[0], f"x must have the same number of features as the domain. Expected {domain.shape[0]}, got {x.shape[1]}."
        return codomain[:, 0] + M * (x - domain[:, 0])

    return f

class Optimizer(ABC):
    """
    Abstract class for optimization algorithms
    """
    @abstractmethod
    def __init__(self, growth_rate: float, population_size, ranges, iterations=None, rng_seed=0):
        pass

    @abstractmethod
    def view(self, x, linmap):
        pass

    @abstractmethod
    def view_g(self):
        pass

    @abstractmethod
    def inverse_view(self, x, linmap=None):
        pass

    @abstractmethod
    def ask_oracle(self) -> np.ndarray:
        raise NotImplementedError("The oracle must be implemented.")

    @abstractmethod
    def init_oracle(self):
        warnings.warn("The oracle initialization is not implemented.")
        pass

    @abstractmethod
    def step(self):
        pass