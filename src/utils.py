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

def calc_freq(frame_length, rate):
    """Function to calculate the frequencies"""
    
    freq = np.fft.rfftfreq(frame_length, 1/rate)
    return freq

def calc_time(frame_length, n_samples, frame_step, rate):
    """Function to calculate the time of each frame (mid-point)"""

    time = np.arange(frame_length/2, n_samples - frame_length/2 + 1, frame_step) / float(rate)
    return time

def draw_specgram(stft):
    """Draw Spectrogram using stft"""

    # Dim: (freq_bins, n_frames)
    if len(stft.shape) == 3:
        stft = stft[0]

    # plt.pcolormesh(time, freq, np.abs(stft), cmap=plt.cm.afmhot)
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    #breakpoint()
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
    cax = ax.matshow(
        np.abs(stft),
        interpolation="nearest",
        aspect="auto",
        cmap=plt.cm.afmhot,
        origin="lower",
    )
    fig.colorbar(cax)
    plt.title("Original Spectrogram")
    plt.show()

def pretty_spectrogram(stft, log=True, thresh=4):
    
    specgram = np.abs(stft)
    if len(specgram.shape) == 3:
        specgram = specgram[0]
    if log == True:
        specgram /= specgram.max()  # volume normalize to max 1
        specgram = np.log10(specgram)
        specgram[specgram < -thresh] = -thresh  # set anything less than the threshold as the threshold
    else:
        specgram[specgram < thresh] = thresh  # set anything less than the threshold as the threshold

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
    cax = ax.matshow(
        specgram,
        interpolation="nearest",
        aspect="auto",
        cmap=plt.cm.afmhot,
        origin="lower",
    )
    fig.colorbar(cax)
    plt.title("Pretty Spectrogram")
    plt.show()

