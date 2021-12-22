from gapy.gapy2 import GA
from pyduino2 import ReactorManager
import numpy as np
from functools import partial

def parse_dados(X,param):
    """
    Parses data from `ReactorManager.log_dados` into an array.

    Args:
        X (:obj:`dict` of :obj:`OrderedDict`): Data obtained from `ReactorManager.log_dados`.
        param (str): Parameter name to be extracted from `X`.
    """
    return np.array(list(map(lambda x: x[1][param],sorted(X.items(),key=lambda x: x[0])))).astype(float)

class genetic_algorithm(ReactorManager,GA):
    def __init__(self,f_param,**kwargs):
        """
        Args:
            f_param (str): Parameter name to be extracted from `ReactorManager.log_dados`.
        """
        ReactorManager.__init__(self)
        GA.__init__(self,population_size=len(self.reactors),**kwargs)
        self.log_init()
        self.f_get = partial(parse_dados,param=f_param)



if __name__ == "__main__":
    g = genetic_algorithm(f_param='DensidadeAtual',mutation_probability=0.01,generations=100,resolution=64,ranges=[[0,1]],elitism=False)