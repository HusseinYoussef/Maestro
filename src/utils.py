import os
import torch
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import json

def audio_info(audio_path):
    """Function to return the info of an audio file without reading the file"""
    
    info = sf.info(audio_path)

    return {
        'samplerate': info.samplerate,
        'samples' : int(info.duration*info.samplerate),
        'channels' : info.channels,
        'duration' : info.duration
    }

def audio_loader(audio_path, start=0, dur=None, always_stereo= True):
    """Read audio file"""

    # info = audio_info(audio_path)
    sample_rate = 44100
    start = int(start * sample_rate)
    if dur:
        stop = start + int(dur * sample_rate)
    else:
        # Read Full file
        stop = None

    audio, rate = sf.read(audio_path, start=start, stop=stop, always_2d=True)  # audio.shape = (samples, channels)
    audio = audio.T # (channels, samples)

    if audio.shape[0] == 1 and always_stereo: # mono.
        audio = np.reshape(audio, audio.shape[1])
        audio = np.array([audio, audio]) # (2, samples)

    # shape (channels, samples)
    return audio, rate

def save_checkpoint(checkpoint_dict:dict, best:bool, target:str, path:str, name:str):
    """Save model checkpoint in the output folder"""
    
    # Save only weights of the model
    if best:
        torch.save(checkpoint_dict['model_state'], os.path.join(path, target+f'_{name}'+'.pth'))

    #Save full checkpoint to resume training
    torch.save(checkpoint_dict, os.path.join(path, target+f'_{name}'+'.chkpnt'))

def learning_curve(path):
    """Draw the learning curve"""

    with open(path, 'r') as fin:
        data = json.load(fin)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(data['train_losses'], label='train', color='r')
    ax.plot(data['valid_losses'], label='valid', color='b')
    ax.scatter(data['best_epoch']-1, data['best_loss'],
            s=500, marker='v', color='c')
    plt.legend(prop={'size': 19})
    plt.ylim([0.0, 0.3])
    ax.set_ylabel('mse loss', size=19)
    ax.set_xlabel('epochs', size=19)
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.show()

def freq_to_bin(max_freq, rate, n_fft):
    """Convert from max_freq to freq_bin"""
    
    bin_width = rate / n_fft
    max_bin = int(max_freq / bin_width) + 1

    return max_bin
    
def convert_to_wav(mp3_path, wav_path, rate=44100, channels=2):
    """Function that takes path to .mp3 audio and converts it to .wav audio"""
    
    if os.path.exists(mp3_path):
        os.system(f'ffmpeg -i "{mp3_path}" -vn -acodec pcm_s16le -ac {channels} -ar {rate} -f wav "{wav_path}"')
    else:
        print("Mp3 file doesn't exist")

def convert_to_mp3(wav_path, mp3_path, rate=44100, channels=2):
    """Function that takes path to .wav audio and converts it to .mp3 audio"""
    
    if os.path.exists(wav_path):
        os.system(f'ffmpeg -i "{wav_path}" -vn -ar {rate} -ac {channels} -b:a 192k "{mp3_path}"')
    else:
        print("Wav file doesn't exist")

def calc_freq(frame_length, rate):
    """Function to calculate the frequencies"""
    
    freq = np.fft.rfftfreq(frame_length, 1/rate)
    return freq

def calc_time(frame_length, n_samples, frame_step, rate):
    """Function to calculate the time of each frame (mid-point)"""

    time = np.arange(frame_length/2, n_samples - frame_length/2 + 1, frame_step) / float(rate)
    return time

def plot_signal(signal, rate=44100, title='Time-domain Signal'):
    """Function to plot time domain signal

    Input Signal of shape: channels, samples
    """
    
    Time = np.linspace(0, signal.shape[1] / rate, num=signal.shape[1])

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(Time, signal[0], 'tomato')
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Amplitudes", fontsize=12)
    plt.title(f'{title}', fontsize=12)
    plt.show()

def calc_fft(signal, rate=44100, title='Frequency Domain'):
    """Function to plot the signal in frequency domain"""

    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1/rate)
    mag = abs(np.fft.rfft(signal)/n)
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(freq, mag, 'tomato')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12) 
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Magnitude (normalized)', fontsize=14)
    plt.title(f'{title}', fontsize=14)
    # plt.show()

def pretty_spectrogram(specgram, log=True, thresh=4, gray=False ,title='Spectrogram'):
    """Function to plot spectrogram"""

    if specgram.ndim == 3:
        specgram = specgram[0]
    elif specgram.ndim == 4:
        specgram = specgram[0][0]

    if log == True:
        specgram /= specgram.max()  # volume normalize to max 1
        specgram = np.log10(specgram)
        specgram[specgram < -thresh] = -thresh  # set anything less than the threshold as the threshold
    else:
        specgram[specgram < thresh] = thresh  # set anything less than the threshold as the threshold

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
    if gray:
        cax = ax.matshow(
            specgram,
            interpolation="nearest",
            aspect="auto",
            cmap=plt.cm.gray,
            origin="lower",
        )
    else:
        cax = ax.matshow(
            specgram,
            interpolation="nearest",
            aspect="auto",
            cmap=plt.cm.inferno,
            origin="lower",
        )
    fig.colorbar(cax)
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12) 
    plt.yticks(np.arange(0, specgram.shape[0]+1, 150))
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Frequency (Hz)", fontsize=14)
    plt.title(f"{title}", fontsize=14)
    plt.show()
