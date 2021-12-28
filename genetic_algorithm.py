from gapy.gapy2 import GA
from pyduino2 import ReactorManager
import numpy as np
from functools import partial
import json
import os
import sched, time

#Path to spectrum.json
SPECTRUM_PATH = "spectrum.json"

s = sched.scheduler(time.time, time.sleep)

def parse_dados(X,param):
    """
    Parses data from `ReactorManager.log_dados` into an array.

    Args:
        X (:obj:`dict` of :obj:`OrderedDict`): Data obtained from `ReactorManager.log_dados`.
        param (str): Parameter name to be extracted from `X`.
    """
    return np.array(list(map(lambda x: x[1][param],sorted(X.items(),key=lambda x: x[0])))).astype(float)

class GeneticAlgorithm(ReactorManager,GA):
    def __init__(self,f_param,**kwargs):
        """
        Args:
            f_param (str): Parameter name to be extracted from `ReactorManager.log_dados`.
        """
        ReactorManager.__init__(self)
        GA.__init__(self,population_size=len(self.reactors),**kwargs)
        self.log_init()
        self.f_get = partial(parse_dados,param=f_param)

        assert os.path.exists(SPECTRUM_PATH)
        with open(SPECTRUM_PATH) as jfile:
            self.spectrum = json.loads(jfile.read())
    def F_get(self):
        """
        Extracts relevant data from Arduinos.
        """
        return self.f_get(self.log_dados())
    def F_set(self,data):
        """
        Sets parameters to Arduinos.
        Args:
            data (:obj:`dict` of :obj:`dict`): Dictionary having reactor id as keys
            and a dictionary of parameters and their values as values.
        """
        for _id,params in data.items():
            self.reactors[self._id[_id]].set(params)
    def F(self,data,delay):
        """
        F_set followed by F_get after a delay.
        """
        F_set(data)
        s.enter(delay,1,lambda x: self.__setattr__("payload",self.F_get()))
        return self.payload
    def set_spectrum(self,preset):
        """
        Sets all reactors with a preset spectrum contained in `SPECTRUM_PATH`.
        """
        command = f"set({','.join(map(lambda x: f'{x[0]},{x[1]}',self.spectrum[preset].items()))})"
        self.send(command,await_response=False)

if __name__ == "__main__":
    g = GeneticAlgorithm(f_param='DensidadeAtual',mutation_probability=0.01,generations=100,resolution=64,ranges=[[0,1]],elitism=False)