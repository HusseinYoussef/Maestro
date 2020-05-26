import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
from scipy import signal
from utils import calc_freq, calc_time, pretty_spectrogram
import matplotlib.pyplot as plt
from utils import audio_loader
import torch
import librosa

class STFT:

    def __init__(
        self,
        frame_length=4096,
        frame_step=1024,
        padding=False,
        center=False,
        pad_mode='reflect',
        scale=False
    ):
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.padding = padding
        self.center = center
        self.pad_mode = pad_mode
        self.window = signal.get_window('hann', self.frame_length)
        self.scale = 1
        if scale:
            self.scale = np.sqrt(1 / self.window.sum()**2)

    def __str__(self):
        
        print(f'n_fft: {self.frame_length}')
        print(f'n_hop: {self.frame_step}')
        return ""

    def __call__(self, audio):
        """
        Input  dims: (batch_size, n_channels, samples)
        Output dims: (batch_size, n_channels, freq_bins, n_frames)
        """

        batch_size, n_channels, signal_length = audio.shape

        assert signal_length > self.frame_length, 'Audio is too short'

        if self.center:
            n_frames = int(signal_length // self.frame_step + 1)
        elif self.padding:
            # Use ceil in case of padding the signal
            n_frames = int(np.ceil(float(np.abs(signal_length - self.frame_length)) / self.frame_step) + 1)
        else:
            n_frames = int(np.abs(signal_length - self.frame_length) // self.frame_step + 1)

        # Pad siganl with reflection on each side
        if self.center:
            new_batch = []
            for sample in range(batch_size):
                new_sample = []
                for ch in range(n_channels):
                    new_sample.append(np.pad(audio[sample][ch],
                                    pad_width=int(self.frame_length // 2), mode=self.pad_mode)) 
            
                new_batch.append(np.stack(new_sample, axis=0))
            audio = np.stack(new_batch, axis=0)
            
        # Pad signal with zeros to produce integer number of frames
        elif self.padding:
            pad_signal_length = (n_frames-1) * self.frame_step + self.frame_length

            assert pad_signal_length >= signal_length, 'Invalid Padding'

            z = torch.zeros((audio.shape[0], audio.shape[1], pad_signal_length-signal_length))
            audio = np.append(audio, z, axis=2)

        indices = np.tile(np.arange(0, self.frame_length), (n_frames, 1)) \
                            + np.tile(np.arange(0, n_frames * self.frame_step, self.frame_step), (self.frame_length, 1)).T

        samples_frames = []
        for sample in audio:
            # Sterro
            if n_channels == 2:
                frames = np.stack((sample[0][indices.astype(np.int32, copy=False)],
                                    sample[1][indices.astype(np.int32, copy=False)]), axis=0)
            # Mono
            else:
                frames = np.array([sample[0][indices.astype(np.int32, copy=False)]])

            # breakpoint()
            frames = self.window * frames

            # frames.shape -> (n_channels, n_frames, frame_length)
            samples_frames.append(frames)

        # batch_size, channels, n_frames, frame_length
        samples_frames = np.array(samples_frames)
        
        # stft = np.fft.rfft(samples_frames, self.frame_length) * self.scale
        
        # swap frames and bins to ease drawing
        # stft = np.transpose(stft, (0, 1, 3, 2))

        return np.transpose(np.fft.rfft(samples_frames, self.frame_length) * self.scale, (0, 1, 3, 2))

class Spectrogram:

    def __init__(self, mono=False, log=False):
        """Convert channels to mono if True"""        
        self.mono = mono
        self.log = log

    def __call__(self, stft):
        """
        I/O dims: (batch_size, n_channels, freq_bins, n_frames)
        """
        # Magnitude
        specgram = np.abs(stft)

        if self.mono:
            specgram = np.mean(specgram, axis=1, keepdims=True)
        if self.log:
            specgram = np.log10(specgram)

        return specgram

def calc_spectrogram(signal):
    """Function to compute spectrogram for plotting"""

    audio = signal[None, ...]
    stft_obj = STFT()
    spec_obj = Spectrogram(mono=True)

    return spec_obj(stft_obj(audio))


if __name__ == "__main__":

    data, rate = audio_loader('F:/CMP 2020/GP/Datasets/Maestro/train/A Classic Education - NightOwl/mixture.wav')
    data = data[:, :8000]

    # mix = torch.stack(sources, dim=0).sum(0)
    # er = torch.nn.functional.mse_loss(mixture, mix)
    # data = data[:,:5000]
    sam = [data, data]
    sam = np.array(sam)

    obj = STFT(center=True)
    sam = data[None, ...]
    res = np.squeeze(obj(sam))
    # inv_obj = iSTFT()
    breakpoint()

    import umx
    unmix = umx.OpenUnmix(
            input_mean=None,
            input_scale=None,
            nb_channels=2,
            hidden_size=512,
            n_fft=4096,
            n_hop=1024,
            max_bin=1487,
            sample_rate=44100
    )
    unmix.stft.center = True
    # breakpoint()
    # ff = calc_freq(4096, rate)
    ch1 = np.squeeze(data[0])
    ch2 = np.squeeze(data[1])
    lib1 = librosa.stft(np.asfortranarray(ch1), n_fft=4096, hop_length=1024, center=True)
    lib2 = librosa.stft(np.asfortranarray(ch2), n_fft=4096, hop_length=1024, center=True)
    lib = np.stack((lib1, lib2), axis=0)
    tor = unmix.transform(torch.from_numpy(sam).float()).permute(1, 2, 3, 0)
    breakpoint()
    er = torch.nn.functional.mse_loss(torch.from_numpy(np.abs(res)).float(), torch.from_numpy(np.abs(lib)).float())
    f, t, zxx = signal.stft(data, fs=rate, window='hann', nperseg=4096, noverlap=4096-1024,boundary=None)
    pretty_spectrogram(res)
    # f, t, Sxx = signal.spectrogram(data, rate, nperseg=4096, noverlap=4096-1024)
    # if zxx.shape[0] == 2:
    #     f2 = plt.figure(2)
    #     plt.pcolormesh(t, f, np.abs(zxx), cmap=plt.cm.afmhot)
    #     plt.title('STFT Magnitudesci')
    #     plt.ylabel('Frequency [Hz]')
    #     plt.xlabel('Time [sec]')
    plt.show()

