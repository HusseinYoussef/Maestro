import numpy as np
import soundfile as sf
from scipy import signal
from utils import draw_specgram, calc_freq, calc_time, pretty_spectrogram
import matplotlib.pyplot as plt

# TODO Support Boundry
class STFT:

    def __init__(self, frame_length=4096, frame_step=1024, padding=True):
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.padding = padding
        self.window = signal.get_window('hann', self.frame_length)
        self.scale = np.sqrt(1 / self.window.sum()**2)

    def __call__(self, audio, rate=44100):
        """
        Input  dims: (batch_size, n_channels, samples)
        Output dims: (batch_size, n_channels, freq_bins, n_frames)
        """

        batch_size, n_channels, signal_length = audio.shape

        # n_frames = ceil((signal_length - frame_length)/step_size)+1
        if self.padding:
            n_frames = int(np.ceil(float(np.abs(signal_length - self.frame_length)) / self.frame_step) + 1)
        else:
            n_frames = np.abs(signal_length - self.frame_length) // self.frame_step + 1
        if signal_length < self.frame_length:
            n_frames = 1

        assert n_frames >= 1, 'Invalid number of frames'

        # Pad signal with zeros to produce integer number of frames
        if self.padding:
            pad_signal_length = (n_frames-1) * self.frame_step + self.frame_length

            assert pad_signal_length >= signal_length, 'Invalid Padding'

            z = np.zeros((audio.shape[0], audio.shape[1], pad_signal_length-signal_length))
            audio = np.append(audio, z, axis=2)

        indices = np.tile(np.arange(0, self.frame_length), (n_frames, 1)) \
                            + np.tile(np.arange(0, n_frames * self.frame_step, self.frame_step), (self.frame_length, 1)).T

        samples_frames = []
        for sample in audio:
            # Sterro
            if n_channels == 2:
                frames = np.array([sample[0][indices.astype(np.int32, copy=False)],
                                    sample[1][indices.astype(np.int32, copy=False)]])
            # Mono
            else:
                frames = np.array([sample[0][indices.astype(np.int32, copy=False)]])

            frames = self.window * frames

            # frame.shape -> (n_frames, frame_length)
            samples_frames.append(frames)

        # batch_size, channels, n_frames, frame_length
        samples_frames = np.array(samples_frames)
        
        stft = np.fft.rfft(samples_frames, self.frame_length) * self.scale
        
        # swap frames and bins to ease drawing
        stft = np.transpose(stft, (0, 1, 3, 2))

        # 1 sec -> rate, ? sec -> sample  tarfeen f wasteen
        time = calc_time(frame_length=self.frame_length, n_samples=audio.shape[-1], frame_step=self.frame_step, rate=rate)
        freq = calc_freq(frame_length=self.frame_length, rate=rate)
        return freq, time, stft

class Spectrogram:

    def __init__(self, mono=False):
        """Convert channels to mono if True"""        
        self.mono = mono

    def __call__(self, stft):
        """
        Input  dims: (batch_size, n_channels, freq_bins, n_frames)
        Output dims: (batch_size, n_frames, n_channesl, freq_bins)
        """

        # Magnitude
        specgram = np.abs(stft)

        if self.mono:
            specgram = np.mean(specgram, axis=1, keepdims=True)

        return specgram


if __name__ == "__main__":

    data, rate = sf.read('mixture.wav')
    if len(data.shape) == 1:
        data = np.reshape(data, (1,-1))
        # data = np.tile(data, (2, 1))
        # data[1] *= 5
    else:
        data = data.T

    # data = data[:, :4*rate]
    # sam = [data1, data1]
    # sam = np.array(sam)

    # breakpoint()
    # obj = STFT()
    # freq, time, res = obj(sam)

    ff = calc_freq(4096, rate)
    #breakpoint()
    f, t, zxx = signal.stft(data, fs=rate, window='hann', nperseg=2048, noverlap=2048-128,boundary=None)
    draw_specgram(zxx)
    pretty_spectrogram(zxx)
    # f, t, Sxx = signal.spectrogram(data, rate, nperseg=4096, noverlap=4096-1024)
    if zxx.shape[0] == 2:
        f2 = plt.figure(2)
        plt.pcolormesh(t, f, np.abs(zxx), cmap=plt.cm.afmhot)
        plt.title('STFT Magnitudesci')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
    plt.show()

