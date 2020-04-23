import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
from torch import Tensor

import soundfile as sf
from utils import draw_specgram,calc_freq,pretty_spectrogram
from feature_extraction import STFT, Spectrogram

from scipy import signal
if __name__ == "__main__":

    data, rate = sf.read('vocals.wav')
    if len(data.shape) == 1:
        data = np.reshape(data, (1,-1))
        # data = np.tile(data, (2, 1))
        # data[1] *= 5
    else:
        data = data.T
    ff = calc_freq(4096, rate)
    f, t, zxx = signal.stft(data, fs=rate, window='hann', nperseg=2048, return_onesided=True, noverlap=2048-128,boundary=None)

    #input= torch.from_numpy(zxx.real)

