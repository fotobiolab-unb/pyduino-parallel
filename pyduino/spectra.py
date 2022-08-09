from gapy.gapy2 import GA
from pyduino.pyduino2 import ReactorManager, chunks, SYSTEM_PARAMETERS, INITIAL_STATE_PATH, REACTOR_PARAMETERS, RELEVANT_PARAMETERS, SCHEMA
import numpy as np
from functools import partial
import json
import os
import pandas as pd
import numpy as np
from typing import Union
import time
from datetime import date, datetime
from pyduino.data_parser import yaml_genetic_algorithm, RangeParser, get_datetimes
from collections import OrderedDict
from scipy.special import softmax
from pyduino.utils import yaml_get, bcolors, TriangleWave, ReLUP
from pyduino.log import datetime_to_str
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
        self.titled_parameters = list(map(lambda x: x.title(),self.parameters))

        RangeParser.__init__(self,ranges,self.parameters)

        #assert os.path.exists(IRRADIANCE_PATH)
        self.irradiance = SYSTEM_PARAMETERS['irradiance']#yaml_get(IRRADIANCE_PATH)
        #self.irradiance = np.array([self.irradiance[u] for u in self.keyed_ranges.keys()])
        self.irradiance = pd.Series(self.irradiance)

        ReactorManager.__init__(self)
        GA.__init__(
            self,
            population_size=len(self.reactors),
            ranges=self.ranges_as_list(),
            generations=0,
            **kwargs
            )
        self.ids = list(self.reactors.keys())
        self.sorted_ids = sorted(self.ids)
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
        f_1 = x_1.loc[self.density_param].astype(float)
        self.power = ((pd.DataFrame(self.G_as_keyed()).T*self.irradiance).sum(axis=1))/100
        if (self.dt is not np.nan) and (self.iteration_counter>0):
            f_0 = x_0.loc[self.density_param].astype(float)
            #self.growth_rate = (f_1-f_0)/self.dt
            self.growth_rate = (f_1/f_0-1)/self.dt
            #self.efficiency = self.growth_rate/(self.power+1)
            self.efficiency = self.growth_rate/(self.power)
        else:
            self.growth_rate = self.power*np.nan
            self.efficiency = self.power*np.nan
        x_1.loc['power',:] = self.power.copy()
        x_1.loc['efficiency',:] = self.efficiency.copy()
        x_1.loc['growth_rate',:] = self.growth_rate.copy()
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
    
    def update_fitness(self,X):
        #Get and return parameter chosen for fitness
        self.fitness = ((-1)**(1+self.maximize))*X.loc[self.fparam].astype(float).to_numpy()
        return self.fitness
    
    def GET(self,tag):
        """
        Collects data from Arduinos and logs it to disk.
        """
        print("[INFO]","GET",datetime.now().strftime("%c"))
        self.past_data = self.data.copy() if self.data is not None else pd.DataFrame(self.payload)
        self.data = pd.DataFrame(self.F_get())
        self.f_map(self.data,self.past_data)
        self.log.log_many_rows(self.data,tags={'growth_state':tag})
        self.log.log_optimal(column=self.fparam,maximum=self.maximize,tags={'growth_state':tag})   
        self.log.log_average(tags={'growth_state':tag})   

    def gotod(self,deltaTgotod):
        self.t_gotod_1 = datetime.now()
        self.send("gotod",await_response=False)
        print("[INFO] gotod sent")
        time.sleep(deltaTgotod)
        self.dt = (datetime.now()-self.t_gotod_1).total_seconds()
        print("[INFO] gotod DT", self.dt)
        self.GET("gotod")

    def run(
        self,
        deltaT: int,
        run_ga: bool = True,
        deltaTgotod: int = None
        ):
        """
        Runs reading and wiriting operations in an infinite loop on intervals given by `deltaT`.

        Args:
            deltaT (int): Amount of time in seconds to wait in each iteration.
            run_ga (bool): Whether or not execute a step in the genetic algorithm.
            deltaTgotod (int, optional): Time to wait after sending `gotod` command.
        """

        #Checking if gotod time is at least five minutes
        if run_ga and deltaTgotod is None: raise ValueError("deltaTgotod must be at least 5 minutes.")
        if run_ga and deltaTgotod <= 5*60: raise ValueError("deltaTgotod must be at least 5 minutes.")

        self.iteration_counter = 1

        self.GET("growing")

        with open("error_traceback.log","w") as log_file:
            log_file.write(datetime_to_str(self.log.timestamp)+'\n')
            try:
                self.deltaT = deltaT
                print("START")
                while True:
                    #growing
                    self.t_grow_1 = datetime.now()
                    time.sleep(max(2,deltaT))
                    self.dt = (datetime.now()-self.t_grow_1).total_seconds()
                    print("[INFO]","DT",self.dt)
                    self.GET("growing")
                    self.update_fitness(self.data)
                    #GA
                    if run_ga:
                        #self.p = softmax(self.fitness/100)
                        self.p = ReLUP(self.fitness*self.fitness*self.fitness)
                        #Hotfix for elitism
                        print(f"{bcolors.OKCYAN}self.data{bcolors.ENDC}")
                        self.data.loc['p',:] = self.p.copy()
                        print(f"{bcolors.BOLD}{self.data.T.loc[:,self.titled_parameters+['power','efficiency','growth_rate','p']]}{bcolors.ENDC}")
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
                        df = self.data.T
                        df.columns = df.columns.str.lower()
                        self.payload = df[self.parameters].T.to_dict()
                        self.G = self.inverse_view(self.payload_to_matrix()).astype(int)
                    print("[INFO]","SET",datetime.now().strftime("%c"))
                    self.F_set(self.payload) if run_ga else None
                    #gotod
                    if self.do_gotod:
                        self.gotod(deltaTgotod)
                    self.iteration_counter += 1
            except Exception as e:
                traceback.print_exc(file=log_file)
                raise(e)

    def run_incremental(
            self,
            deltaT: int,
            parameter: str,
            deltaTgotod: int = None,
            N: int = 1,
            M: int = 1,
            bounds:list = [100,0]
            ):
        """
        Runs reading and wiriting operations in an infinite loop on intervals given by `deltaT` and increments parameters
        periodically on an interval given by `deltaClockHours`.

        Args:
            deltaT (int): Amount of time in seconds to wait in each iteration.
            parameter (str): Name of the parameters to be updated.
            deltaTgotod (int, optional): Time to wait after sending `gotod` command.
            N (int): Number of iteration groups to wait to trigger a parameter update.
            M (int): Number of iterations to wait to increment `N`.
            bounds: Starts on `bounds[0]` and goes towards `bounds[1]`. Then, the other way around.
        """

        #Initialize stepping
        df = pd.DataFrame(self.payload).T
        self.triangles = list(map(lambda x: TriangleWave(x,bounds[0],bounds[1],N),df.loc[:,parameter].to_list()))
        self.triangle_wave_state = 1 if bounds[1] >= bounds[0] else -1
        c = 0
        m = 0

        #Checking if gotod time is at least five minutes
        if deltaTgotod is not None and deltaTgotod < 5*60: print(bcolors.WARNING,"[WARNING]","deltaTgotod should be at least 5 minutes.",bcolors.ENDC)

        with open("error_traceback.log","w") as log_file:
            log_file.write(datetime_to_str(self.log.timestamp)+'\n')
            try:
                self.deltaT = deltaT
                while True:
                    self.t1 = datetime.now()
                    self.GET("growing")
                    self.update_fitness(self.data)
                    #gotod
                    if self.do_gotod:
                        self.send("gotod",await_response=False)
                        print("[INFO] gotod sent")
                        time.sleep(deltaTgotod)
                        self.dt = (datetime.now()-self.t1).total_seconds()
                        print("[INFO] gotod DT", self.dt)
                        self.GET("gotod")
                        self.t1 = datetime.now()
                    
                    # Pick up original parameters from preset state and increment them with `parameter_increment`.
                    df = pd.DataFrame(self.data).T
                    df.loc[:,parameter] = list(map(lambda T: T.y(c),self.triangles))
                    print("[INFO]","WAVE","UP" if self.triangle_wave_state > 0 else "DOWN", "COUNTER", str(m), "LEVEL", str(c))
                    if c%N==0:
                        self.triangle_wave_state *= -1
                    # ---------------------------

                    df.columns = df.columns.str.lower()
                    self.payload = df[self.parameters].T.to_dict()
                    self.G = self.inverse_view(self.payload_to_matrix()).astype(int)
                    print("[INFO]","SET",self.t1.strftime("%c"))
                    self.F_set(self.payload)
                    time.sleep(max(2,deltaT))
                    m+=1
                    if (m%M)==0:
                        c += 1
                    self.t2 = datetime.now()
                    self.dt = (self.t2-self.t1).total_seconds()
                    print("[INFO]","DT",self.dt)
            except Exception as e:
                traceback.print_exc(file=log_file)
                raise(e)


if __name__ == "__main__":
    g = Spectra(**hyperparameters)
