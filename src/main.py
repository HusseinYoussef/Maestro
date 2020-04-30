'''  This file for only testing functions from seperate files together '''
import utils
import soundfile as sf
import numpy as np
from dataset import expand_track, split_track

if __name__ == "__main__":

    track_path = 'D:/CMP/4th/GP/Datasets[OUT]/musdb18/train/A Classic Education - NightOwl/mixture.wav'
    track, sample_rate = sf.read(track_path, always_2d=True)
    split_track(track, sample_rate, 5, 'result', '1')

    '''
    mag, rate = utils.audio_loader('D:/CMP/4th/GP/Datasets[OUT]/musdb18/train/A Classic Education - NightOwl/mixture.wav')
    mag = mag.T
    length = len(mag[0]) / 44100
    samples = 5 * 44100
    sf.write('new_file.wav', mag, rate)
    '''