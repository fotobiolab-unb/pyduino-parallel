import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyduino.utils import get_param, partition
import numpy as np
import dotenv
dotenv.load_dotenv()
dotenv.load_dotenv(dotenv_path=".env.local")

def test_get_param():
    # Test case 1: No IDs provided
    data = {'A': {'param1': 1, 'param2': 2}, 'B': {'param1': 3, 'param2': 4}}
    key = 'param1'
    expected_output = {'A': 1, 'B': 3}
    assert get_param(data, key) == expected_output

    # Test case 2: IDs provided
    data = {'A': {'param1': 1, 'param2': 2}, 'B': {'param1': 3, 'param2': 4}}
    key = 'param1'
    ids = {'A'}
    expected_output = {'A': 1}
    assert get_param(data, key, ids) == expected_output

    # Test case 5: Check order of keys with IDs provided
    data = {'A': {'param1': 1, 'param2': 2}, 'B': {'param1': 3, 'param2': 4}, 'C': {'param1': 5, 'param2': 6}}
    key = 'param1'
    ids = {'C', 'A'}
    expected_output = {'C': 5, 'A': 1}
    assert get_param(data, key, ids) == expected_output
    
def test_partition():
    # Test case 1: Array size is divisible by n
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    n = 2
    expected_output = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    assert np.array_equal(partition(X, n), expected_output)

    # Test case 2: Array size is not divisible by n
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    n = 3
    expected_output = [np.array([[1, 2], [3, 4], [5, 6]]), np.array([[7, 8], [9, 10]])]
    assert all(np.array_equal(o, e) for o,e in zip(partition(X, n), expected_output))

    # Test case 3: Array size is smaller than n
    X = np.array([[1, 2]])
    n = 3
    expected_output = [np.array([[1, 2]])]
    assert np.array_equal(partition(X, n), expected_output)