from gapy.gapy2 import GA
from pyduino2 import ReactorManager
import numpy as np
from functools import partial
import json
import os
import pandas as pd
import time
from datetime import datetime

#Path to spectrum.json
SPECTRUM_PATH = "spectrum.json"

#Path to relevant parameters file
PARAM_PATH = "relevant_parameters.txt"

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

class GeneticAlgorithm(ReactorManager,GA):
    def __init__(self,f_param,log_name=None,**kwargs):
        """
        Args:
            f_param (str): Parameter name to be extracted from `ReactorManager.log_dados`.
        """
        ReactorManager.__init__(self)
        GA.__init__(self,population_size=len(self.reactors),**kwargs)
        self.log_init(name=log_name)
        self.f_get = partial(parse_dados,param=f_param)

        assert os.path.exists(SPECTRUM_PATH)
        with open(SPECTRUM_PATH) as jfile:
            self.spectrum = json.loads(jfile.read())
        
        assert os.path.exists(PARAM_PATH)
        with open(PARAM_PATH) as txt:
            self.parameters = list(map(lambda x: x.strip(),txt.readlines()))
        
        self.payload = None
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
            print("[INFO]","GET",datetime.now().strftime("%c"))
            self.data = self.F_get()
            if run_ga:
                pass
            time.sleep(deltaT)
            print("[INFO]","SET",datetime.now().strftime("%c"))
            print(pd.DataFrame(row_subset(self.data,self.parameters)))
            self.F_set(row_subset(self.data,self.parameters))
            time.sleep(2)
            self.send("quiet_connect",await_response=False)
    def calibrate(self,deltaT=120,dir="calibrate"):
        """
        Runs `curva` and dumps the result into txts.
        """
        if not os.path.exists(dir):
            os.mkdir(dir)
        out = {}
        self.send("curva",await_response=False)
        time.sleep(deltaT)
        for name,reactor in self.reactors.items():
            out[name] = reactor._conn.read_until('*** fim da curva dos LEDs ***'.encode('ascii'))
            with open(os.path.join(dir,f"reator_{self._id_reverse[name]}.txt"),"w") as f:
                f.write(out[name].decode('ascii'))
        return out

if __name__ == "__main__":
    g = GeneticAlgorithm(log_name="20211229113606",f_param='DensidadeAtual',mutation_probability=0.01,generations=100,resolution=64,ranges=[[0,1]],elitism=False)