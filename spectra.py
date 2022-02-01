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
from data_parser import yaml_genetic_algorithm, RangeParser, get_datetimes
from collections import OrderedDict
from scipy.special import softmax

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

#Path to spectrum.json
SPECTRUM_PATH = os.path.join(__location__,"spectrum.json")

#Path to relevant parameters file
PARAM_PATH = os.path.join(__location__,"relevant_parameters.txt")

#Path to hyperparameters for the genetic algorithm
HYPER_PARAM = os.path.join(__location__,"hyperparameters.yaml")

def update_dict(D,A,key):
    """
    Updates dictionary `D` with values in `A` with key name `key`.
    
    Args:
        D (:obj:dict of :obj:dict): Input dictionary.
        A (dict): Dict of values to be added with the same keys as in `D`.
        key (str): Key name to assign to values of `A`.
    """
    for k,v in A.items():
        D[k].update({key:v})


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

def seval(x):
    try:
        return eval(x)
    except:
        return x

def parse_dados(X,param):
    """
    Parses data from `ReactorManager.log_dados` into an array while attempting to convert the input string into an appropriate data type.

    Args:
        X (:obj:`dict` of :obj:`OrderedDict`): Data obtained from `ReactorManager.log_dados`.
        param (str): Parameter name to be extracted from `X`.
    """
    return np.array(list(map(seval,map(lambda x: x[1][param],sorted(X.items(),key=lambda x: x[0])))))

class Spectra(RangeParser,ReactorManager,GA):
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
        self.fitness = np.nan * np.ones(len(self.reactors))
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
            self.power = self.view(self.G,self.linmap).sum(axis=1)
            #print("[DEBUG]","f_1-f_0",f_1-f_0)
            self.density = (f_1 - f_0)/self.dt
            F = self.density/self.power
            F[F == np.inf] == 0
            F = np.nan_to_num(F)
            return F
        else:
            self.density = np.nan * np.ones(len(self.reactors))
            self.power = np.nan * np.ones(len(self.reactors))
            return np.zeros_like(f_1)

    def F_get(self):
        """
        Extracts relevant data from Arduinos.
        """
        return self.dados()
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
        self.deltaT = deltaT
        while True:
            self.t1 = datetime.now()
            print("[INFO]","GET",datetime.now().strftime("%c"))
            self.past_data = self.data.copy() if self.data is not None else None
            self.data = self.F_get()
            self.fitness = self.f_map(self.data,self.past_data)
            if run_ga:
                self.p = softmax(self.fitness)
                self.crossover()
                self.mutation()
                self.payload = self.G_as_keyed()
            else:
                df = pd.DataFrame(self.data).T
                df.columns = df.columns.str.lower()
                self.payload = df[self.parameters].T.to_dict()
            update_dict(self.data,dict(zip(self._id.keys(),self.fitness)),'fitness')
            update_dict(self.data,dict(zip(self._id.keys(),self.power)),'power')
            update_dict(self.data,dict(zip(self._id.keys(),self.density)),'density')
            self.log.log_many_rows(self.data)
            print("[INFO]","SET",self.t1.strftime("%c"))
            self.F_set(self.payload) if run_ga else None
            time.sleep(2)
            time.sleep(deltaT)
            self.send("quiet_connect",await_response=False)
            self.t2 = datetime.now()
            self.dt = (self.t2-self.t1).total_seconds()
            print("[INFO]","DT",self.dt)

if __name__ == "__main__":
    hyperparameters = yaml_genetic_algorithm(HYPER_PARAM)
    g = Spectra(**hyperparameters)