# Tests the linear map.
import sys
import os
import numpy as np
import dotenv
dotenv.load_dotenv()
dotenv.load_dotenv(dotenv_path=".env.local")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyduino.optimization import linmap

def test_linmap():
    domain = [(0,1), (0,1), (0,1)]
    codomain = [(-1,1), (-10,10), (0,8)]
    f = linmap(domain, codomain)
    assert f(np.random.random((10,3))).shape == (10,3)