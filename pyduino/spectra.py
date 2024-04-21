from pyduino.optimization.nelder_mead import NelderMeadBounded
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
from pyduino.utils import bcolors, get_param
from pyduino.paths import PATHS
from pyduino.log import datetime_to_str, y_to_table, to_markdown_table
import traceback
import warnings
import threading
from datetime import datetime
import logging

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

#Path to spectrum.json
SPECTRUM_PATH = os.path.join(__location__,"spectrum.json")
#Path to irradiance values
IRRADIANCE_PATH = os.path.join(__location__,"irradiance.yaml")

logging.basicConfig(
    filename='pyduino.log',
    filemode='w',
    level=PATHS.SYSTEM_PARAMETERS.get('log_level', logging.INFO),
)


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

class Spectra(RangeParser,ReactorManager,NelderMeadBounded):
    def __init__(self,
        ranges,
        density_param,
        brilho_param=None,
        maximize=True,
        log_name=None,
        reset_density=False,
        **kwargs
        ):
        """
        Initializes the Spectra class.

        Args:
            ranges (dict): A dictionary of parameters with a two-element list containing the minimum and maximum attainable values for each parameter.
            density_param (str): The name of the parameter to be used as the density count.
            brilho_param (float, optional): When nonzero, it will be used to turn optimization on or off. Defaults to None.
            maximize (bool, optional): Whether to maximize the fitness function. Defaults to True.
            log_name (str, optional): The name of the log. Defaults to None.
            reset_density (bool, optional): Whether to reset density values on the reactors at each iteration. Defaults to False.
            **kwargs: Additional keyword arguments.

        Attributes:
            spectrum (dict): A dictionary containing the spectrum data.
            parameters (list): A list of relevant system parameters.
            titled_parameters (list): A list of relevant system parameters with their titles capitalized.
            irradiance (str): The irradiance value.
            ids (list): A list of reactor IDs.
            sorted_ids (list): A sorted list of reactor IDs.
            tensorboard_path (str): The path to the tensorboard log.
            writer (SummaryWriter): The tensorboard summary writer.
            payload (dict): The payload data.
            data (None): Placeholder for data.
            do_gotod (bool): Whether to reset density values.
            dt (float): The time step value.

        Raises:
            AssertionError: If the spectrum file does not exist.

        """
        assert os.path.exists(SPECTRUM_PATH)
        with open(SPECTRUM_PATH) as jfile:
            self.spectrum = json.loads(jfile.read())
        
        self.parameters = PATHS.SYSTEM_PARAMETERS['relevant_parameters']
        self.titled_parameters = list(map(lambda x: x.title(),self.parameters))
        RangeParser.__init__(self,ranges,self.parameters)

        self.irradiance = PATHS.SYSTEM_PARAMETERS['irradiance']

        ReactorManager.__init__(self, kwargs.get('include', None))
        NelderMeadBounded.__init__(
            self,
            population_size=len(self.parameters),
            ranges=self.ranges_as_list(),
            rng_seed=kwargs.get('rng_seed',0)
        )
        self.sorted_ids = sorted(self.ids)
        self.log_init(name=log_name)
        self.tensorboard_path = os.path.join(self.log.prefix, "runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.writer = SummaryWriter(self.tensorboard_path)
        logging.info(f"{bcolors.OKGREEN}Created tensorboard log at {self.tensorboard_path}{bcolors.ENDC}")
        self.payload = self.population_as_dict if self.payload is None else self.payload
        self.data = None
        self.do_gotod = reset_density
        self.density_param = density_param
        self.brilho_param = brilho_param
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

    def log_data(self, i, tags={}):
        """
        Logs the tensor values and fitness scores.

        This method iterates over the tensor values and fitness scores and logs them using the writer object.
        """
        logging.info(f"LOGGING {datetime.now().strftime('%c')}")
        data = self.F_get()
        additional_parameters = {}
        if PATHS.TENSORBOARD is not None and "additional_parameters" in PATHS.TENSORBOARD:
            additional_parameters_source = PATHS.TENSORBOARD["additional_parameters"]
            for param in additional_parameters_source:
                additional_parameters[param] = get_param(data, param, self.reactors.keys())

        logging.debug(f"ADDITIONAL PARAMETERS {additional_parameters}")
        
        P = self.view_g()
        #Log main parameters
        for k,(rid, ry) in enumerate(self.y.items()):
            self.writer.add_scalar(f'reactor_{rid}/y', float(ry), i)
            for r_param_id, rparam in enumerate(self.parameters):
                self.writer.add_scalar(f'reactor_{rid}/{rparam}', float(P[k][r_param_id]), i)
            for param, value in additional_parameters.items():
                self.writer.add_scalar(f'reactor_{rid}/{param}', float(value[rid]), i)
        if self.maximize:
            self.writer.add_scalar('optima', max(self.y), i)
        else:
            self.writer.add_scalar('optima', min(self.y), i)

        # Log the DataFrame as a table in text format
        self.writer.add_text("reactor_state", text_string=to_markdown_table(data), global_step=i)

        self.log.log_many_rows(data,tags=tags)

    def gotod(self):
        self.t_gotod_1 = datetime.now()
        self.send("gotod",await_response=False)
        logging.debug(f"gotod sent")
        time.sleep(self.deltaTgotod)
        self.dt = (datetime.now()-self.t_gotod_1).total_seconds()
        logging.debug(f"gotod DT {self.dt}")

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
            deltaTgotod (int, optional): The time interval for performing optimization. Defaults to None.

        Notes:
            - If mode is 'optimize' and deltaTgotod is less than or equal to 300, a warning will be raised.
            - If mode is 'free', the number of rows in X must be equal to the number of reactors.

        """
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.kill()
        self.thread = threading.Thread(target=self._run, args=(deltaT, mode, deltaTgotod))
        self.thread.start()
    
    def _run(self, deltaT: float, mode: str = 'optimize', deltaTgotod: int = None):
        # Checking if gotod time is at least five minutes
        if mode == "optimize" and isinstance(deltaTgotod, int) and deltaTgotod <= 300:
            warnings.warn("deltaTgotod should be at least 5 minutes.")

        if mode == "free":
            assert self.population.shape[0] == len(self.reactors), "X must have the same number of rows as reactors in free mode."

        self.deltaT = deltaT
        self.deltaTgotod = deltaTgotod
        self.iteration_counter = 1

        with open("error_traceback.log", "w") as log_file:
            log_file.write(datetime_to_str(self.log.timestamp) + '\n')
            try:
                logging.debug("START")
                while True:
                    # growing
                    self.t_grow_1 = datetime.now()
                    time.sleep(max(2, deltaT))
                    self.dt = (datetime.now() - self.t_grow_1).total_seconds()
                    logging.debug(f"DT {self.dt}")
                    # Optimizer
                    if mode == "optimize":
                        if self.brilho_param is None:
                            self.step()
                        else:
                            brilhos = np.array(list(get_param(self.F_get(), self.brilho_param, self.reactors)))
                            if np.all(brilhos > 0):
                                self.step()
                            else:
                                logging.info(f"{self.brilho_param} is off. No optimization steps are being performed.")
                        if isinstance(self.deltaTgotod, int):
                            self.gotod()
                    elif mode == "free":
                        data = self.F_get()
                        self.y = get_param(data, self.density_param, self.reactors)
                    logging.debug(f"SET {datetime.now().strftime('%c')}")
                    print(y_to_table(self.y))
                    self.log_data(self.iteration_counter)
                    self.iteration_counter += 1
            except Exception as e:
                traceback.print_exc(file=log_file)
                raise(e)

class SpectraManager():
    def __init__(self, g:dict):
        self.g = g

    def call_method(self, method, *args, **kwargs):
        for s in self.g.values():
            getattr(s, method)(*args, **kwargs)
    
    def get_attr(self, attr):
        return {k:getattr(v, attr) for k,v in self.g.items()}
    
    def run(self, deltaT, mode, deltaTgotod):
        for s in self.g.values():
            s.run(deltaT, mode, deltaTgotod)
    
    @property
    def reactors(self):
        reactors = {}
        for s in self.g.values():
            reactors.update(s.reactors)
        return reactors

    def __repr__(self) -> str:
        rstr = bcolors.BOLD + bcolors.OKGREEN + "Main Manager\n" + bcolors.ENDC
        for v in self.g.values():
            rstr += f"├── {repr(v)}" + "\n"
        return rstr


if __name__ == "__main__":
    g = Spectra(**PATHS.HYPERPARAMETERS)
