import os
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def audio_info(audio_path):
    """Function to return the info of an audio file without reading the file"""
    
    info = sf.info(audio_path)

    return {
        'samplerate': info.samplerate,
        'samples' : int(info.duration*info.samplerate),
        'channels' : info.channels,
        'duration' : info.duration
    }

def audio_loader(audio_path, start=0, dur=None):
    """Read audio file"""

    info = audio_info(audio_path)
    start = int(start * info['samplerate'])
    if dur:
        stop = start + int(dur * info['samplerate'])
    else:
        # Read Full file
        stop = None

    audio, rate = sf.read(audio_path, start=start, stop=stop, always_2d=True)

    # shape (channels, samples)
    return audio.T, rate

def freq_to_bin(max_freq, rate, n_fft):
    """Convert from max_freq to freq_bin"""
    
    bin_width = rate / n_fft
    max_bin = int(max_freq / bin_width) + 1

    return max_bin
    
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
    breakpoint()
    
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
    # plt.show()

def pretty_spectrogram(stft, log=True, thresh=4):
    
    specgram = np.abs(stft)
    if len(specgram.shape) == 3:
        specgram = specgram[0]
    if len(specgram.shape) == 4:
        specgram = specgram[0][0]
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
    # plt.show()

