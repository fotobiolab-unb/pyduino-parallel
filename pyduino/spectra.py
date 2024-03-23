from pyduino.optimization.nelder_mead import NelderMead
from pyduino.pyduino2 import ReactorManager, chunks, PATHS
import numpy as np
import json
import os
import pandas as pd
import numpy as np
import time
from datetime import date, datetime
from pyduino.data_parser import RangeParser
from collections import OrderedDict
from pyduino.utils import yaml_get, bcolors, TriangleWave, get_param
from pyduino.log import datetime_to_str
import traceback

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

#Path to spectrum.json
SPECTRUM_PATH = os.path.join(__location__,"spectrum.json")

#Path to hyperparameters for the genetic algorithm
hyperparameters = PATHS.HYPERPARAMETERS

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

class Spectra(RangeParser,ReactorManager,NelderMead):
    def __init__(self,elitism,ranges,density_param,maximize=True,log_name=None,reset_density=False,**kwargs):
        """
        Args:
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
        self.parameters = PATHS.SYSTEM_PARAMETERS['relevant_parameters']#yaml_get(PARAM_PATH)
        self.titled_parameters = list(map(lambda x: x.title(),self.parameters))

        RangeParser.__init__(self,ranges,self.parameters)

        self.irradiance = PATHS.SYSTEM_PARAMETERS['irradiance']#yaml_get(IRRADIANCE_PATH)
        self.irradiance = pd.Series(self.irradiance)

        ReactorManager.__init__(self)
        NelderMead.__init__(
            self,
            population_size=len(self.reactors),
            ranges=self.ranges_as_list(),
            rng_seed=kwargs.get('rng_seed',0)
        )
        self.ids = list(self.reactors.keys())
        self.sorted_ids = sorted(self.ids)
        self.log_init(name=log_name)        
        self.payload = self.population_as_dict if self.payload is None else self.payload
        self.data = None
        self.do_gotod = reset_density
        self.density_param = density_param
        self.maximize = maximize
        self.dt = np.nan
        self.elitism = elitism
    def assign_to_reactors(self, x):
        """
        Assigns a list of parameters to the reactors.

        Parameters:
        x (list): The input list to be converted.

        Returns:
        OrderedDict: An ordered dictionary where the keys are the IDs and the values are the ranges.

        """
        ids = self.ids[:len(x)]
        return OrderedDict(
            zip(
                ids,
                map(
                    lambda u: self.ranges_as_keyed(u),
                    list(np.round(self.view(x),2))
                )
            )
        )
    @property
    def population_as_dict(self):
        """
        Converts genome matrix into an appropriate format to send to the reactors.
        """
        return self.assign_to_reactors(self.population)
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
        df.loc['fitness'] = self.y
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
    def init(self):
        """
        Sets payload to the reactors.
        """
        self.F_set(self.payload)

    def set_spectrum(self,preset):
        """
        Sets all reactors with a preset spectrum contained in `SPECTRUM_PATH`.
        """
        command = f"set({','.join(map(lambda x: f'{x[0]},{x[1]}',self.spectrum[preset].items()))})"
        self.send(command,await_response=False)

    def set_preset_state_spectra(self,*args,**kwargs):
        self.set_preset_state(*args,**kwargs)
        self.population = self.inverse_view(self.payload_to_matrix()).astype(int)   
    
    def GET(self,tag):
        """
        Collects data from Arduinos and logs it to disk.
        """
        print("[INFO]","GET",datetime.now().strftime("%c"))
        self.past_data = self.data.copy() if self.data is not None else pd.DataFrame(self.payload)
        self.data = self.F_get()
        self.log.log_many_rows(pd.DataFrame(self.data),tags={'growth_state':tag})
        self.log.log_optimal(column=self.density_param,maximum=self.maximize,tags={'growth_state':tag})   
        self.log.log_average(tags={'growth_state':tag}, cols=[self.density_param])   

    def gotod(self):
        self.t_gotod_1 = datetime.now()
        self.send("gotod",await_response=False)
        print("[INFO] gotod sent")
        time.sleep(self.deltaTgotod)
        self.dt = (datetime.now()-self.t_gotod_1).total_seconds()
        print("[INFO] gotod DT", self.dt)

    # === Optimizer methods ===
    def ask_oracle(self, X) -> np.ndarray:
        """
        Asks the oracle for the fitness of the given input.

        Parameters:
        X (np.ndarray): The input for which the fitness is to be calculated. Must be already mapped to codomain.

        Returns:
        np.ndarray: The fitness value calculated by the oracle.
        """
        y = np.array([])

        assert X.shape[1] == len(self.parameters)
        assert len(X.shape) == 2, "X must be a 2D array."
        n_partitions = len(X) // len(self.reactors) + (len(X) % len(self.reactors) > 0)
        partitions = np.array_split(X, n_partitions)

        for partition in partitions:
            payload = self.assign_to_reactors(partition)
            reactors = payload.keys()

            self.gotod()
            data0 = self.F_get()
            f0 = get_param(data0, self.density_param, reactors)
            f0 = np.array(list(f0.values())).astype(float)

            self.F_set(payload)
            time.sleep(self.deltaT)
            data = self.F_get()
            f = get_param(data, self.density_param, reactors)
            f = np.array(list(f.values())).astype(float)

            alpha = (np.log(f) - np.log(f0))/self.deltaT #Growth Rate $f=f_0 exp(alpha T)$

            y = np.append(y,alpha)
        return -y   
    # === * ===

    def run(
        self,
        deltaT: float,
        run_optim: bool = True,
        deltaTgotod: int = None
        ):
        """
        Runs reading and wiriting operations in an infinite loop on intervals given by `deltaT`.

        Args:
            deltaT (float): Amount of time in seconds to wait in each iteration.
            run_optim (bool): Whether or not to use the optimizer.
            deltaTgotod (int, optional): Time to wait after sending `gotod` command.
        """

        #Checking if gotod time is at least five minutes
        if run_optim and (deltaTgotod is None or deltaTgotod <= 300): raise ValueError("deltaTgotod must be at least 5 minutes.")

        self.deltaT = deltaT
        self.deltaTgotod = deltaTgotod
        self.iteration_counter = 1
        self.GET("growing")

        with open("error_traceback.log","w") as log_file:
            log_file.write(datetime_to_str(self.log.timestamp)+'\n')
            try:
                print("START")
                while True:
                    #growing
                    self.t_grow_1 = datetime.now()
                    time.sleep(max(2,deltaT))
                    self.dt = (datetime.now()-self.t_grow_1).total_seconds()
                    print("[INFO]","DT",self.dt)
                    self.GET("growing")
                    #Optimizer
                    if run_optim:
                        self.step()
                    self.gotod()
                    print("[INFO]","SET",datetime.now().strftime("%c"))
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
        OBSOLETE

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
                    self.population = self.inverse_view(self.payload_to_matrix()).astype(int)
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
