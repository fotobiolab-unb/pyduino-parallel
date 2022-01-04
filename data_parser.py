import yaml
from collections import OrderedDict

def yaml_genetic_algorithm(filename):
    """
    Loads hyperparameters for the genetic algorithm from a YAML file.
    """
    y = None
    with open(filename) as f:
        y = yaml.load(f.read(),yaml.Loader)
    return y

class RangeParser:
    def __init__(self,ranges,parameter_list):
        """
        Args:
            ranges (:obj:dict of :obj:list): Dictionary of parameters with a two element list containing the
                its minimum and maximum attainable values respectively.
            parameter_list (list): List of parameter names.
        """
        self.keyed_ranges = OrderedDict(ranges)
        if 'others' in self.keyed_ranges.keys():
            for p in list(set(parameter_list)-self.keyed_ranges.keys()):
                self.keyed_ranges[p] = self.keyed_ranges['others']
            del self.keyed_ranges['others']

    def ranges_as_keyed(self,x=None):
        """
        Converts range list into keyed format.
        """
        if x is None:
            return self.keyed_ranges
        else:
            return OrderedDict(zip(self.keyed_ranges.keys(),x))
    def ranges_as_list(self,x=None):
        """
        Converts range dictionary into list format.
        """
        if x is None:
            return self.ranges_as_list(self.keyed_ranges)
        else:
            return list(x.values())