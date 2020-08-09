import multiprocessing as mp
import sharedmem as shm
import numpy as np
import matplotlib.pyplot as plt 
import random
import time
from functools import reduce
import utils
from featurizer import STFT, Spectrogram, calc_spectrogram
from Pool import Pool


if __name__ == "__main__":
    
    #utils.convert_to_wav('D:/CMP/4th/GP/Test/Cheap.mp3', 'D:/CMP/4th/GP/Test/Cheap.wav')
    #utils.convert_to_mp3('D:/CMP/4th/GP/Test/B[stem].wav', 'D:/CMP/4th/GP/Test/Sia [Vocals].mp3')
    '''
    drums_track,_ = utils.audio_loader('D:/CMP/4th/GP/Test/Buitraker - Revo X/drums.wav')
    vocal_track,_ = utils.audio_loader('D:/CMP/4th/GP/Test/Buitraker - Revo X/vocals.wav')


    utils.pretty_spectrogram(calc_spectrogram(drums_track), title='drums only')
    utils.pretty_spectrogram(calc_spectrogram(vocal_track), title='vocal only')
    '''
    MP = dict()
    MP['hassan'] = 'osama'
    for x in MP:
        print(x)

    
    #print(manager.go(sq, X))