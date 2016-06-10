import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

from pfm import *

from PIL import Image

from concurrent.futures import ThreadPoolExecutor

import os.path
import glob

import random

class Dataset( object ):
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def shuffle(self):
        indices = list(range(len(self.inputs)))
        shuffled_indices = random.shuffle(indices)
        self.inputs = [self.inputs[i] for i in shuffled_indices]
        self.outputs = [self.outputs[i] for i in shuffled_indices]

class Trainer( object ):
    def __init__(self):
        self.training_function = None
        self.dataset = []
    
