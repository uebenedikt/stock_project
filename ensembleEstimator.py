"""
Creates and trains the ensemble estimator. There are 40 NN of 6 different architectures involved in the estimation of
the next price.
"""
import pathlib
import os
import string
import datetime
import random
import numpy as np
import math
import multiprocessing as mp
import pickle
import json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import tensorflow.python.framework.errors_impl as errors
from sklearn.linear_model import SGDRegressor


def exists(today):
    """

    """
    check = False
    path = pathlib.Path('.').joinpath('ensembleEstimator').joinpath(str(today))
    try:
        with open(path.joinpath('architecture_map.json'), 'r') as file:
            d = json.load(file)
    except FileNotFoundError:
        pass
    else:
        print(f'Found an ensemble of size {len(d)}.')
        check = True
    return check
