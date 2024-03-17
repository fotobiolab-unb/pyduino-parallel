import numpy as np
from typing import Callable
from . import Optimizer, linmap

class NelderMead(Optimizer):
    def __init__(self, population_size: int, ranges: list[float], rng_seed: int = 0):
        """
        This Nelder-Mead algorithm assumes that the optimization function 'f' is to be minimized and time independent.

        ranges list of pairs:
            Maxima and minima that each parameter can cover.
            Example: [(0, 10), (0, 10)] for two parameters ranging from 0 to 10.
        population_size int:
            Number of individuals in the population.
        rng_seed int:
            Seed for the random number generator. Default is 0.
        """
        self.population_size = population_size
        self.ranges = np.array(ranges)
        self.rng_seed = np.random.default_rng(rng_seed)

        # Derived attributes
        self.invlinmap = linmap(self.ranges, np.array([[0, 1]] * len(self.ranges)))
        self.linmap = linmap(np.array([[0, 1]] * len(self.ranges)), self.ranges)

        # Initialize the population (random position and initial momentum)
        self.population = self.rng_seed.random((self.population_size, len(self.ranges)))

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
    
    def ask_oracle(self, X: np.ndarray) -> np.ndarray:
        return super().ask_oracle()

    def init_oracle(self):
        return self.ask_oracle(self.view_g())

    def step(self):
        """
        This function performs a single step of the Nelder-Mead algorithm.
        """
        
        # Sort the population by the value of the oracle
        self.y = self.ask_oracle(self.view_g())
        idx = np.argsort(self.y)
        self.population = self.population[idx]

        # Compute the centroid of the population
        centroid = self.population[:-1].mean(axis=0)

        # Reflect the worst point through the centroid
        reflected = centroid + (centroid - self.population[-1])

        # Evaluate the reflected point
        
        y_reflected = self.ask_oracle(self.view(reflected.reshape(1,-1)))

        # If the reflected point is better than the second worst, but not better than the best, then expand
        
        if y_reflected < self.y[-2] and y_reflected > self.y[0]:
            expanded = centroid + (reflected - centroid)
            y_expanded = self.ask_oracle(self.view(expanded.reshape(1,-1)))
            if y_expanded < y_reflected:
                self.population[-1] = expanded
            else:
                self.population[-1] = reflected
        # If the reflected point is better than the best, then expand
        elif y_reflected < self.y[0]:
            expanded = centroid + 2 * (reflected - centroid)
            y_expanded = self.ask_oracle(self.view(expanded.reshape(1,-1)))
            if y_expanded < y_reflected:
                self.population[-1] = expanded
            else:
                self.population[-1] = reflected
        # If the reflected point is worse than the second worst, then contract
        elif y_reflected > self.y[-2]:
            contracted = centroid + 0.5 * (self.population[-1] - centroid)
            y_contracted = self.ask_oracle(self.view(contracted.reshape(1,-1)))
            if y_contracted < self.y[-1]:
                self.population[-1] = contracted
            else:
                for i in range(1, len(self.population)):
                    self.population[i] = 0.5 * (self.population[i] + self.population[0])
        # If the reflected point is worse than the worst, then shrink
        elif y_reflected > self.y[-1]:
            for i in range(1, len(self.population)):
                self.population[i] = 0.5 * (self.population[i] + self.population[0])

        return self.view_g()