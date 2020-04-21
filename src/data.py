import numpy as np


class augmentation:

    def __init__(self, functions:list):
        self.functions = functions

    def __call__(self, audio):
        for fn in self.functions:
            audio = fn(audio)
        return audio

    def gain(self, audio, low=0.25, high=1.25):
        """Apply random gain to the signal between low and high"""
        g = low + np.random.rand() * (high-low)
        return audio * g

    def channel_swap(self, audio):
        """Swap channels of stero audio with a probabilty of 0.5"""
        if audio.shape[0] == 2 and np.random.rand() >= 0.5:
            audio[[0,1]] = audio[[1,0]]
        else:
            return audio
