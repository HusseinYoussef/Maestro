import numpy as np 
from functools import reduce
import math
import utils
import soundfile as sf
from featurizer import STFT, Spectrogram
from glob import glob
import os
import time
from tqdm import tqdm
import reconstruction
import multiprocessing as mp
import time

def run(x, y):
    return x*y

def wrapper(inp):
    return run(*inp)

if __name__ == "__main__":
    
    with mp.Pool(mp.cpu_count()) as p:
        print(p.map(wrapper, [(1,2), (2, 3), (3, 4)]))
