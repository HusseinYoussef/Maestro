import os
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def audio_info(audio_path):
    """Function to retrieve the info of an audio file"""
    
    data, rate = sf.read(audio_path)

    fname = os.path.basename(audio_path).split('.')[0]
    samples = data.shape[0]
    duration = data.shape[0] // rate
    channels = data.shape[1]

    return {
        'name' : fname,
        'n_samples' : samples,
        'n_channels' : channels,
        'duration' : duration
    }

def calc_freq(nfft, rate):
    """Function to calculate the frequencies"""
    
    freq = np.fft.rfftfreq(nfft, 1/rate)
    return freq

def calc_time(nfft, n_samples, nhop, rate):
    """Function to calculate the time of each frame (mid-point)"""

    time = np.arange(nfft/2, n_samples - nfft/2 + 1, nhop) / float(rate)
    return time

def draw_stft(freq, time, stft):

    if len(stft.shape == 3):
        stft = stft[0]

    plt.pcolormesh(time, freq, np.abs(stft), cmap=plt.cm.afmhot)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

