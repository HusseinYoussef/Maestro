import multiprocessing as mp
import sharedmem as shm
import numpy as np
import matplotlib.pyplot as plt 
import random
import time
from functools import reduce
import utils

if __name__ == "__main__":
    
    #utils.convert_to_wav('D:/CMP/4th/GP/Test/Cheap thrills (Sia).mp3', 'D:/CMP/4th/GP/Test/Cheap thrills (Sia).wav')
    #utils.convert_to_mp3('D:/CMP/4th/GP/Test/B[stem].wav', 'D:/CMP/4th/GP/Test/Sia [Vocals].mp3')

    drums_track,_ = utils.audio_loader('D:/CMP/4th/GP/Test/Buitraker - Revo X/drums.wav')
    vocal_track,_ = utils.audio_loader('D:/CMP/4th/GP/Test/Buitraker - Revo X/vocals.wav')

    utils.plot_signal(drums_track, title= 'time domain [drums]')
    utils.plot_signal(vocal_track, title= 'time domain [vocals]')



 