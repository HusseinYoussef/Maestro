import numpy as np
import torch
import torch.nn.functional as F
from featurizer import STFT, Spectrogram
from utils import freq_to_bin

class UMX(torch.nn.Module):

    def __init__(
        self, 
        frame_length=4096,
        frame_step=1024,
        rate=44100,
        n_channels=2,
        n_hidden=512,
        input_mean=None,
        input_scale=None,
        max_freq=None,
        output_norm=False,
        n_layer=3,
        device='cpu'
    ):
        super().__init__()

        # For cropping the signal (downsampling)
        self.output_bins = frame_length // 2 + 1
        if max_freq and 0 < max_freq <= rate/2:
            # Bin 0:    -312.5 Hz  to    312.5 Hz  (center:     0.0 Hz)
            # Bin 1:     312.5 Hz  to    937.5 Hz  (center:   625.0 Hz)
            # Bin 2:     937.5 Hz  to   1562.5 Hz  (center:  1250.0 Hz)
            self.freq_bins = freq_to_bin(max_freq=max_freq, rate=rate, n_fft=frame_length)
        else:
            self.freq_bins = self.output_bins
        
        self.stft = STFT(frame_length=frame_length, frame_step=frame_step, padding=True)
        self.specgram = Spectrogram(mono=(n_channels==1))
        self.n_hidden = n_hidden
        self.output_norm = output_norm
        self.device = device
        
        # 1st fc layer
        self.fc1 = torch.nn.Linear(
            in_features=self.freq_bins*n_channels,
            out_features=n_hidden,
            bias=False
        )

        # Apply Batch normalization
        self.bn1 = torch.nn.BatchNorm1d(num_features=self.fc1.out_features)

        # Stacked LSTM
        # input (batch, seq_len, feat)
        # output (batch, seq_len, direction*hidden)
        assert n_layer >= 1, 'Invalid number of LSTM layers'
        self.lstm = torch.nn.LSTM(
            input_size=self.n_hidden,
            hidden_size=self.n_hidden // 2,
            num_layers=n_layer,
            bias=True,
            batch_first=True,
            dropout=0.4,
            bidirectional=True
        )

        # 2nd fc layer
        self.fc2 = torch.nn.Linear(
            in_features=self.n_hidden*2,
            out_features=self.n_hidden,
            bias=False
        )

        self.bn2 = torch.nn.BatchNorm1d(num_features=self.fc2.out_features)

        # 3rd fc layer
        self.fc3 = torch.nn.Linear(
            in_features=self.n_hidden,
            out_features=self.output_bins*n_channels,
            bias=False
        )

        self.bn3 = torch.nn.BatchNorm1d(num_features=self.fc3.out_features)

        if input_mean is None:
            input_mean = torch.zeros(self.freq_bins).to(device)
        else:
            input_mean = torch.from_numpy(input_mean[:self.freq_bins]).float().to(device)

        if input_scale is None:
            input_scale = torch.ones(self.freq_bins).to(device)
        else:
            input_scale = torch.from_numpy(1.0/input_scale[:self.freq_bins]).float().to(device)

        self.input_mean = torch.nn.Parameter(input_mean)
        self.input_scale = torch.nn.Parameter(input_scale)
        self.output_mean = torch.nn.Parameter(torch.ones(self.output_bins).float(), requires_grad=self.output_norm)
        self.output_scale = torch.nn.Parameter(torch.ones(self.output_bins).float(), requires_grad=self.output_norm)

    def extract_features(self, x):
        """extract features of a given input of shape (batch_size, channels, samples)"""
        return (torch.from_numpy(self.specgram(self.stft(x))).float()).permute(0,3,1,2)

    def forward(self, x):
        """move input throught the model's pipeline"""

        # x = self.extract_features(x)
        # x = x.to(self.device)
        # x shape (batch_size, frames, n_channels, freq_bins)
        batch_size, frames, n_channels, freq_bins = x.shape

        mix = x.clone()

        # Downsampling / crop the high frequencies
        x = x[..., :self.freq_bins]
        
        # Normalize input
        x -= self.input_mean
        x *= self.input_scale

        x = self.fc1(x.reshape(batch_size*frames, n_channels*self.freq_bins))
        x = self.bn1(x)
        # x shape (batch_size*frames, hidden_size)
        x = x.reshape(batch_size, frames, self.n_hidden)
        # x shape (batch_size, frames, hidden_size)

        x = torch.tanh(x)

        lstm_out = self.lstm(x)[0]
        # lstm_out shape (batch_size, frames, hidden_size)

        # skip connection
        x = torch.cat([x, lstm_out], dim=-1)
        # x shape (batch_size, frames, 2*hidden_size)

        x = self.fc2(x.reshape(batch_size*frames, x.shape[-1]))
        x = self.bn2(x)
        x = F.relu(x)
        # x shape (batch_size*frames, hidden_size)

        x = self.fc3(x)
        x = self.bn3(x)
        # x shape (batch_size*frames, channels*freq_bins)
        x = x.reshape(batch_size, frames, n_channels, self.output_bins)

        if self.output_norm:
            x *= self.output_scale
            x += self.output_mean

        # original shape
        x = F.relu(x) * mix

        return x


if __name__ == "__main__":
    
    model = UMX(frame_length=4096, frame_step=1024)
    sample = np.random.rand(5, 2, 5000)
    x = torch.zeros((2,3))
    xz = torch.ones((2,3))
    breakpoint()
    sample = torch.tensor(sample)
    model(sample)
    print("Hi")