from pyduino.optimization.nelder_mead import NelderMead
from pyduino.pyduino2 import ReactorManager, chunks, PATHS
import numpy as np
import json
import os
import pandas as pd
import numpy as np
import time
from tensorboardX import SummaryWriter
from datetime import date, datetime
from pyduino.data_parser import RangeParser
from collections import OrderedDict
from pyduino.utils import yaml_get, bcolors, TriangleWave, get_param
from pyduino.log import datetime_to_str
import traceback
import warnings

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
    def __init__(self,ranges,density_param,maximize=True,log_name=None,reset_density=False,**kwargs):
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

        self.irradiance = PATHS.SYSTEM_PARAMETERS['irradiance']

        ReactorManager.__init__(self)
        NelderMead.__init__(
            self,
            population_size=len(self.parameters),
            ranges=self.ranges_as_list(),
            rng_seed=kwargs.get('rng_seed',0)
        )
        self.ids = list(self.reactors.keys())
        self.sorted_ids = sorted(self.ids)
        self.log_init(name=log_name)
        self.writer = SummaryWriter(self.log.prefix)
        print(bcolors.OKGREEN,"[INFO]", "Created tensorboard log at", self.log.prefix, bcolors.ENDC)  
        self.payload = self.population_as_dict if self.payload is None else self.payload
        self.data = None
        self.do_gotod = reset_density
        self.density_param = density_param
        self.maximize = maximize
        self.dt = np.nan
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
    
    @property
    def power(self):
        return {reactor_ids: sum(vals[key]*self.irradiance[key] for key in self.irradiance.keys()) for reactor_ids, vals in self.payload.items()}

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

    def log_tensor(self, i):
        """
        Logs the tensor values and fitness scores.

        This method iterates over the tensor values and fitness scores and logs them using the writer object.
        """
        print(bcolors.BOLD,"[INFO]","LOGGING",datetime.now().strftime("%c"), bcolors.ENDC)
        P = self.view_g()
        for k, v in enumerate(self.y):
            self.writer.add_scalar(f'fitness/{k}', v, i)
            for j, u in enumerate(self.parameters):
                self.writer.add_scalar(f'{u}/{k}', P[k][j], i)
        if self.maximize:
            self.writer.add_scalar('optima', max(self.y), i)
        else:
            self.writer.add_scalar('optima', min(self.y), i)

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
            self.payload = self.assign_to_reactors(partition)
            reactors = self.payload.keys()

            self.gotod()
            data0 = self.F_get()
            f0 = get_param(data0, self.density_param, reactors)

            self.F_set(self.payload)
            time.sleep(self.deltaT)
            data = self.F_get()
            f = get_param(data, self.density_param, reactors)

            #yield_rate = np.array([(float(f[id])/float(f[id]) - 1)/self.deltaT/self.power[id] for id in reactors]).astype(float)
            
            fitness = np.array([self.power[id] for id in reactors]).astype(float)

            y = np.append(y,((-1)**(self.maximize))*(fitness))

        self.y = y
        return y   
    # === * ===

    def run(
        self,
        deltaT: float,
        mode: str = 'optimize',
        deltaTgotod: int = None
    ):
        """
        Run the bioreactor simulation.

        Args:
            deltaT (float): The time step for the simulation.
            mode (str, optional): The mode of operation. Defaults to 'optimize'.
            deltaTgotod (int, optional): The time interval for the 'gotod' operation. Defaults to None.

        Raises:
            ValueError: If deltaTgotod is not an integer.

        Notes:
            - If mode is 'optimize' and deltaTgotod is less than or equal to 300, a warning is raised.
            - If mode is 'free', the number of rows in X must be equal to the number of reactors.

        """
        # Checking if gotod time is at least five minutes
        if isinstance(deltaTgotod, int):
            if mode == "optimize" and deltaTgotod <= 300:
                warnings.warn("deltaTgotod should be at least 5 minutes.")
        else:
            raise ValueError("deltaTgotod must be an integer")

        if mode == "free":
            assert X.shape[0] == len(self.reactors), "X must have the same number of rows as reactors in free mode."

        self.deltaT = deltaT
        self.deltaTgotod = deltaTgotod
        self.iteration_counter = 1

        with open("error_traceback.log", "w") as log_file:
            log_file.write(datetime_to_str(self.log.timestamp) + '\n')
            try:
                print("START")
                while True:
                    # growing
                    self.t_grow_1 = datetime.now()
                    time.sleep(max(2, deltaT))
                    self.dt = (datetime.now() - self.t_grow_1).total_seconds()
                    print("[INFO]", "DT", self.dt)
                    # Optimizer
                    if mode == "optimize":
                        self.step()
                        if isinstance(self.deltaTgotod, int):
                            self.gotod()
                    elif mode == "free":
                        data = self.F_get()
                        self.y = get_param(data, self.density_param, self.reactors)
                    print("[INFO]", "SET", datetime.now().strftime("%c"))
                    self.log_tensor(self.iteration_counter)
                    self.iteration_counter += 1
            except Exception as e:
                traceback.print_exc(file=log_file)
                raise(e)

if __name__ == "__main__":
    g = Spectra(**hyperparameters)
