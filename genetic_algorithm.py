from gapy.gapy2 import GA
from pyduino2 import ReactorManager
import numpy as np
from functools import partial
import json
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from data_parser import yaml_genetic_algorithm, RangeParser
from collections import OrderedDict
from scipy.special import softmax

#Path to spectrum.json
SPECTRUM_PATH = "spectrum.json"

#Path to relevant parameters file
PARAM_PATH = "relevant_parameters.txt"

#Path to hyperparameters for the genetic algorithm
HYPER_PARAM = "hyperparameters.yaml"

def dict_subset(D,keys):
    """
    Extracts a subdictionary from a given dictionary.

    Args:
        D (dict): Input dictionary.
        keys (:obj:list of :obj:str): A list of keys pertaining to the dictionary.
    """
    return {k:D[k] for k in keys if k in D}

def row_subset(rows,keys):
    """
    Extracts a subset from a `rows` table.

    Args: 
        rows (:obj:`dict` of :obj:`OrderedDict`): Data obtained from `ReactorManager.log_dados`.
        keys (:obj:list of :obj:str): A list of keys.
    """
    return pd.DataFrame(rows).T.loc[:,keys].T.astype(float).astype(int).astype(str).to_dict()

def parse_dados(X,param):
    """
    Parses data from `ReactorManager.log_dados` into an array.

    Args:
        X (:obj:`dict` of :obj:`OrderedDict`): Data obtained from `ReactorManager.log_dados`.
        param (str): Parameter name to be extracted from `X`.
    """
    return np.array(list(map(lambda x: x[1][param],sorted(X.items(),key=lambda x: x[0])))).astype(float)

class GeneticAlgorithm(RangeParser,ReactorManager,GA):
    def __init__(self,f_param,ranges,log_name=None,**kwargs):
        """
        Args:
            f_param (str): Parameter name to be extracted from `ReactorManager.log_dados`.
            ranges (:obj:dict of :obj:list): Dictionary of parameters with a two element list containing the
                its minimum and maximum attainable values respectively.
        """

        assert os.path.exists(SPECTRUM_PATH)
        with open(SPECTRUM_PATH) as jfile:
            self.spectrum = json.loads(jfile.read())
        
        assert os.path.exists(PARAM_PATH)
        with open(PARAM_PATH) as txt:
            self.parameters = list(map(lambda x: x.strip(),txt.readlines()))

        RangeParser.__init__(self,ranges,self.parameters)
        ReactorManager.__init__(self)
        GA.__init__(
            self,
            population_size=len(self.reactors),
            ranges=self.ranges_as_list(),
            generations=0,
            **kwargs
            )
        self.log_init(name=log_name)        
        self.payload = self.G_as_keyed()
        self.data = None
        self.fparam = f_param
    def G_as_keyed(self):
        """
        Converts genome matrix into an appropriate format to send to the reactors.
        """
        return OrderedDict(
            zip(
                self._id.keys(),
                map(
                    lambda u: self.ranges_as_keyed(u),
                    list(self.view(self.G,self.linmap).astype(int))
                )
            )
        )
    def f_map(self,x_1,x_0):
        """
        Computation for the fitness function.
        """
        f_1 = partial(parse_dados,param=self.fparam)(x_1)
        if x_0 is not None:
            f_0 = partial(parse_dados,param=self.fparam)(x_0)
            power = self.view(self.G,self.linmap).sum(axis=1)
            F = (f_1 - f_0)/power
            F[F == np.inf] == 0
            F = np.nan_to_num(F)
            return F
        else:
            return np.zeros_like(f_1)

    def F_get(self):
        """
        Extracts relevant data from Arduinos.
        """
        return self.log_dados()
    def F_set(self,data):
        """
        Sets parameters to Arduinos.
        Args:
            data (:obj:`dict` of :obj:`dict`): Dictionary having reactor id as keys
            and a dictionary of parameters and their values as values.
        """
        for _id,params in data.items():
            self.reactors[self._id[_id]].set(params)
    def set_spectrum(self,preset):
        """
        Sets all reactors with a preset spectrum contained in `SPECTRUM_PATH`.
        """
        command = f"set({','.join(map(lambda x: f'{x[0]},{x[1]}',self.spectrum[preset].items()))})"
        self.send(command,await_response=False)
    def run(self,deltaT,run_ga=True):
        """
        Runs reading and wiriting operations in an infinite loop on intervals given by `deltaT`.

        Args:
            deltaT (int): Amount of time in seconds to wait in each iteration.
            run_ga (bool): Whether or not execute a step in the genetic algorithm.
        """
        while True:
            print("[INFO]","SET",datetime.now().strftime("%c"))
            self.F_set(self.payload)
            time.sleep(2)
            time.sleep(deltaT)
            self.send("quiet_connect",await_response=False)
            print("[INFO]","GET",datetime.now().strftime("%c"))
            self.past_data = self.data.copy() if self.data is not None else None
            self.data = self.F_get()
            if run_ga:
                self.fitness = self.f_map(self.data,self.past_data)
                self.p = softmax(self.fitness)
                self.crossover()
                self.mutation()
                self.payload = self.G_as_keyed()
            else:
                df = pd.DataFrame(self.data).T
                df.columns = df.columns.str.lower()
                self.payload = df[self.parameters].T.to_dict()

if __name__ == "__main__":
    hyperparameters = yaml_genetic_algorithm(HYPER_PARAM)
    g = GeneticAlgorithm(**hyperparameters)