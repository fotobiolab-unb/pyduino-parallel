import numpy as np
from typing import Callable, List
from . import Optimizer, linmap

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def logit(x, inf=100):
    return np.where(x==0, -inf, np.where(x==1, inf, np.log(x) - np.log(1-x)))

class NelderMead(Optimizer):
    def __init__(self, population_size: int, ranges: List[float], rng_seed: int = 0, hypercube_radius = 100):
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
        self.a = 1/hypercube_radius
        self.backward = lambda x: logit(linmap(self.ranges, np.array([[0, 1]] * len(self.ranges)))(x))/self.a
        self.forward = lambda x: linmap(np.array([[0, 1]] * len(self.ranges)), self.ranges)(sigmoid(self.a * x))

        # Initialize the population (random position and initial momentum)
        self.population = self.rng_seed.random((self.population_size, len(self.ranges)))
        
        # Initialize y as vector of nans
        self.y = np.full(self.population_size, np.nan)

    def view(self, x):
        """
        Maps the input from the domain to the codomain.
        """
        return self.forward(x)
    
    def view_g(self):
        """
        Maps the input from the domain to the codomain.
        """
        return self.forward(self.population)

    def inverse_view(self, x):
        """
        Maps the input from the codomain to the domain.
        """
        return self.backward(x)
    
    def ask_oracle(self, X: np.ndarray) -> np.ndarray:
        return super().ask_oracle()

    def init_oracle(self):
        return self.ask_oracle(self.view_g())

    def step(self):
        """
        This function performs a single step of the Nelder-Mead algorithm.
        """
        
        # Sort the population by the value of the oracle
        y = self.ask_oracle(self.view_g())
        idx = np.argsort(y)
        self.population = self.population[idx]

        # Compute the centroid of the population
        centroid = self.population[:-1].mean(axis=0)

        # Reflect the worst point through the centroid
        reflected = centroid + (centroid - self.population[-1])

        # Evaluate the reflected point
        
        y_reflected = self.ask_oracle(self.view(reflected.reshape(1,-1)))

        # If the reflected point is better than the second worst, but not better than the best, then expand
        
        if y_reflected < y[-2] and y_reflected > y[0]:
            expanded = centroid + (reflected - centroid)
            y_expanded = self.ask_oracle(self.view(expanded.reshape(1,-1)))
            if y_expanded < y_reflected:
                self.population[-1] = expanded
            else:
                self.population[-1] = reflected
        # If the reflected point is better than the best, then expand
        elif y_reflected < y[0]:
            expanded = centroid + 2 * (reflected - centroid)
            y_expanded = self.ask_oracle(self.view(expanded.reshape(1,-1)))
            if y_expanded < y_reflected:
                self.population[-1] = expanded
            else:
                self.population[-1] = reflected
        # If the reflected point is worse than the second worst, then contract
        elif y_reflected > y[-2]:
            contracted = centroid + 0.5 * (self.population[-1] - centroid)
            y_contracted = self.ask_oracle(self.view(contracted.reshape(1,-1)))
            if y_contracted < y[-1]:
                self.population[-1] = contracted
            else:
                for i in range(1, len(self.population)):
                    self.population[i] = 0.5 * (self.population[i] + self.population[0])
        # If the reflected point is worse than the worst, then shrink
        elif y_reflected > y[-1]:
            for i in range(1, len(self.population)):
                self.population[i] = 0.5 * (self.population[i] + self.population[0])

        self.y = y.copy()
        return self.view_g()