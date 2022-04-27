from gapy.gapy2 import GA
from pyduino2 import ReactorManager, chunks, SYSTEM_PARAMETERS, INITIAL_STATE_PATH, REACTOR_PARAMETERS, RELEVANT_PARAMETERS, SCHEMA
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
from utils import yaml_get, bcolors
from log import datetime_to_str
import traceback

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

#Path to spectrum.json
SPECTRUM_PATH = os.path.join(__location__,"spectrum.json")

#Path to hyperparameters for the genetic algorithm
SYS_PARAM = os.path.join(__location__,"config.yaml")
hyperparameters = yaml_get(SYS_PARAM)['hyperparameters']

#Path to irradiance values
IRRADIANCE_PATH = os.path.join(__location__,"irradiance.yaml")

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
    return pd.DataFrame(rows).T.loc[:,keys].T.astype(float).astype(str).to_dict()

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
    return np.array(list(map(seval,map(lambda x: x[1].get(param,0),sorted(X.items(),key=lambda x: x[0])))))

class Spectra(RangeParser,ReactorManager,GA):
    def __init__(self,elitism,f_param,ranges,density_param,maximize=True,log_name=None,reset_density=False,**kwargs):
        """
        Args:
            f_param (str): Parameter name to be extracted from `ReactorManager.log_dados`.
            ranges (:obj:dict of :obj:list): Dictionary of parameters with a two element list containing the
                its minimum and maximum attainable values respectively.
            reset_density (bool): Whether or not to reset density values on the reactors at each iteration.
            maximize (bool): Whether or not to maximize the fitness function.
            density_param (str): Name of the parameter which will be used as density count.
        """

        assert os.path.exists(SPECTRUM_PATH)
        with open(SPECTRUM_PATH) as jfile:
            self.spectrum = json.loads(jfile.read())
        
        #assert os.path.exists(PARAM_PATH)
        self.parameters = SYSTEM_PARAMETERS['relevant_parameters']#yaml_get(PARAM_PATH)

        RangeParser.__init__(self,ranges,self.parameters)

        #assert os.path.exists(IRRADIANCE_PATH)
        self.irradiance = SYSTEM_PARAMETERS['irradiance']#yaml_get(IRRADIANCE_PATH)
        self.irradiance = np.array([self.irradiance[u] for u in self.keyed_ranges.keys()])

        ReactorManager.__init__(self)
        GA.__init__(
            self,
            population_size=len(self.reactors),
            ranges=self.ranges_as_list(),
            generations=0,
            **kwargs
            )
        self.ids = list(self.reactors.keys())
        self.log_init(name=log_name)        
        self.payload = self.G_as_keyed() if self.payload is None else self.payload
        self.data = None
        self.do_gotod = reset_density
        self.fparam = f_param
        self.density_param = density_param
        self.fitness = np.nan * np.ones(len(self.reactors))
        self.maximize = maximize
        self.dt = np.nan
        self.elitism = elitism
    def G_as_keyed(self):
        """
        Converts genome matrix into an appropriate format to send to the reactors.
        """
        return OrderedDict(
            zip(
                self.ids,
                map(
                    lambda u: self.ranges_as_keyed(u),
                    list(np.round(self.view(self.G,self.linmap),2))
                )
            )
        )
    def f_map(self,x_1,x_0):
        """
        Computation for the fitness function.
        """
        f_1 = partial(parse_dados,param=self.density_param)(x_1).astype(float)
        self.power = (self.view_g()*self.irradiance).sum(axis=1)/100.0
        if self.dt is not np.nan:
            f_0 = partial(parse_dados,param=self.density_param)(x_0).astype(float)
            self.growth_rate = (f_1 - f_0)/self.dt
            self.efficiency = self.growth_rate/self.power
            self.efficiency[self.efficiency == np.inf] == 0
            self.efficiency = np.nan_to_num(self.efficiency)
        else:
            #Use default values if there's no past data
            self.growth_rate = np.zeros(len(self.reactors))
            self.efficiency = np.zeros_like(f_1)
        #Added new columns to current data
        update_dict(x_1,dict(zip(self.ids,self.power)),'power')
        update_dict(x_1,dict(zip(self.ids,self.efficiency)),'efficiency')
        update_dict(x_1,dict(zip(self.ids,self.growth_rate)),'growth_rate')
        #Get and return parameter chosen for fitness
        self.fitness = ((-1)**(1+self.maximize))*pd.DataFrame(x_1).loc[self.fparam].astype(float).to_numpy()
        #self.fitness = 61.1-partial(parse_dados,param=self.fparam)(x_1).astype(float)
        return self.fitness
    def payload_to_matrix(self):
        return np.nan_to_num(
            np.array(
                [[self.payload[i].get(u,np.nan) for u in self.keyed_ranges.keys()] for i in self.ids]
                ).astype(float)
            )
    def data_dict_to_matrix(self,D):
        return np.nan_to_num(
            np.array(
                [[self.D[i].get(u,np.nan) for u in self.parameters] for i in self.ids]
                ).astype(float)
            )
    def pretty_print_dict(self,D):
        df = pd.DataFrame(D)
        df.index = df.index.str.lower()
        df = df.loc[self.parameters,:]
        df.loc['fitness'] = self.fitness
        df.loc['probs'] = 100*self.p
        return df.round(decimals=2)
    def F_get(self):
        """
        Extracts relevant data from Arduinos.
        """
        return self.dados()
    def F_set(self,x):
        """
        Sets parameters to Arduinos.
        Args:
            x (:obj:`dict` of :obj:`dict`): Dictionary having reactor id as keys
            and a dictionary of parameters and their values as values.
        """
        for _id,params in x.items():
            for chk in chunks(list(params.items()),3):
                self.reactors[_id].set(dict(chk))
                time.sleep(1)
    def set_spectrum(self,preset):
        """
        Sets all reactors with a preset spectrum contained in `SPECTRUM_PATH`.
        """
        command = f"set({','.join(map(lambda x: f'{x[0]},{x[1]}',self.spectrum[preset].items()))})"
        self.send(command,await_response=False)

    def set_preset_state_spectra(self,*args,**kwargs):
        self.set_preset_state(*args,**kwargs)
        self.G = self.inverse_view(self.payload_to_matrix()).astype(int)

    def run(self,deltaT,run_ga=True):
        """
        Runs reading and wiriting operations in an infinite loop on intervals given by `deltaT`.

        Args:
            deltaT (int): Amount of time in seconds to wait in each iteration.
            run_ga (bool): Whether or not execute a step in the genetic algorithm.
        """
        with open("error_traceback.log","w") as log_file:
            log_file.write(datetime_to_str(self.log.timestamp)+'\n')
            try:
                self.deltaT = deltaT
                while True:
                    self.t1 = datetime.now()
                    print("[INFO]","GET",datetime.now().strftime("%c"))
                    self.past_data = self.data.copy() if self.data is not None else self.payload
                    self.data = self.F_get()
                    #gotod
                    if self.do_gotod:
                        gotod_response = self.send_parallel("gotod",60,True)
                        gotod_response = list(map(lambda x: json.loads(x[1]),gotod_response))
                        print("[INFO] gotod")
                        print(pd.DataFrame(gotod_response))
                    #---
                    self.f_map(self.data,self.past_data)
                    if run_ga:
                        self.p = softmax(self.fitness)
                        #Hotfix for elitism
                        print(f"{bcolors.OKCYAN}self.data{bcolors.ENDC}")
                        print(f"{bcolors.BOLD}{pd.DataFrame(self.pretty_print_dict(self.data))}{bcolors.ENDC}")
                        if self.elitism:
                            self.elite_ix = self.ids[self.p.argmax()]
                            self.anti_elite_ix = self.ids[self.p.argmin()]
                            self.elite = self.G[self.p.argmax()].copy()
                        self.crossover()
                        self.mutation()
                        if self.elitism:
                            self.G[self.p.argmin()] = self.elite.copy()
                        self.payload = self.G_as_keyed()
                    else:
                        df = pd.DataFrame(self.data).T
                        df.columns = df.columns.str.lower()
                        self.payload = df[self.parameters].T.to_dict()
                        self.G = self.inverse_view(self.payload_to_matrix()).astype(int)
                    self.log.log_many_rows(self.data)
                    self.log.log_optimal(column=self.fparam,maximum=self.maximize)
                    print("[INFO]","SET",self.t1.strftime("%c"))
                    self.F_set(self.payload) if run_ga else None
                    time.sleep(2)
                    time.sleep(deltaT)
                    #self.send("quiet_connect",await_response=False)
                    self.t2 = datetime.now()
                    self.dt = (self.t2-self.t1).total_seconds()
                    print("[INFO]","DT",self.dt)
            except Exception as e:
                traceback.print_exc(file=log_file)
                raise(e)


if __name__ == "__main__":
    g = Spectra(**hyperparameters)
    #g.set_preset_state(path=INITIAL_STATE_PATH)