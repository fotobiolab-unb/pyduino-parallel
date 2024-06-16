import numpy as np
from . import Optimizer, linmap

class GradientDescent(Optimizer):
    def __init__(self, population_size: int, ranges: list[float], damping: float = 0.01, rng_seed: int = 0):
        r"""
        This gradient descent algorithm assumes that the optimization function 'f' is to be minimized, differentiable, and time independent.

        $$\frac{\mathrm{d} f}{\mathrm{d} t} = \frac{\partial f}{\partial \vec{x}}\frac{\mathrm{d} \vec{x}}{\mathrm{d} t}$$

        Where $\frac{\partial f}{\partial t}$ is assumed to be zero.

        Args:
            ranges (list of pairs): Maxima and minima that each parameter can cover.
                Example: [(0, 10), (0, 10)] for two parameters ranging from 0 to 10.
            population_size (int): Number of individuals in the population.
            damping (float, optional): Damping factor to avoid oscillations. Defaults to 0.01.
            rng_seed (int, optional): Seed for the random number generator. Defaults to 0.
        """
        self.population_size = population_size
        self.ranges = np.array(ranges)
        self.damping = damping
        self.rng_seed = np.random.default_rng(rng_seed)

        # Derived attributes
        self.invlinmap = linmap(self.ranges, np.array([[0, 1]] * len(self.ranges)))
        self.linmap = linmap(np.array([[0, 1]] * len(self.ranges)), self.ranges)

        # Initialize the population (random position and initial momentum)
        self.population = self.rng_seed.random((self.population_size, len(self.ranges)))
        self.momenta = self.rng_seed.random((self.population_size, len(self.ranges)))
        self.oracle_past = self.rng_seed.random((self.population_size, 1))

    def view(self, x):
        """
        Maps the input from the domain to the codomain.
        """
        return self.linmap(x)
    
    def view_g(self):
        """
        Maps the input from the domain to the codomain.
        """
        return self.linmap(self.population)

    def inverse_view(self, x):
        """
        Maps the input from the codomain to the domain.
        """
        return self.invlinmap(x)
    
    def ask_oracle(self) -> np.ndarray:
        return super().ask_oracle()
    
    def set_oracle(self, X: np.ndarray):
        return super().set_oracle(X)

    def init_oracle(self):
        return self.set_oracle(self.view(self.population))

    def step(self, dt: float):
        """
        Moves the population in the direction of the gradient.

        dt float:
            Time taken from the last observation.
        """
        pass
