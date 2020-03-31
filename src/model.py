import numpy as np
import torch
from feature_extraction import STFT, Spectrogram
from utils import freq_to_bin

# TODO NORMALIZE input

class UMX(torch.nn.Module):

    def __init__(
        self, 
        frame_length=4096,
        frame_step=1024,
        rate=44100,
        n_channels=2,
        n_hidden=512,
        max_freq=None,
        fc_bias=True,
        n_layer=3
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
        self.fc_bias = fc_bias

        # 1st fc layer
        self.fc1 = torch.nn.Linear(
            in_features=self.freq_bins*n_channels,
            out_features=n_hidden,
            bias=self.fc_bias
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
            bias=self.fc_bias
        )

        self.bn2 = torch.nn.BatchNorm1d(num_features=self.fc2.out_features)

        # 3rd fc layer
        self.fc3 = torch.nn.Linear(
            in_features=self.n_hidden,
            out_features=self.output_bins*n_channels,
            bias=self.fc_bias
        )

        self.bn3 = torch.nn.BatchNorm1d(num_features=self.fc3.out_features)

    def forward(self, x):

        # x shape (batch_size, n_channels, samples)

        x = self.stft(x)[-1]
        breakpoint()
        x = np.transpose(self.specgram(x), (0, 3, 1, 2))

        # x shape (batch_size, frames, n_channels, freq_bins)

        x = torch.Tensor(x)
        mix = x.clone()
        batch_size, frames, n_channels, freq_bins = x.shape

        x = x[..., :self.freq_bins]
        
        x = self.fc1(x.reshape(batch_size*frames, n_channels*freq_bins))
        x = self.bn1(x)
        x = torch.tanh(x)
        # x shape (batch_size*frames, hidden_size)

        x = x.reshape(batch_size, frames, self.n_hidden)
        # x shape (batch_size, frames, hidden_size)

        lstm_out = self.lstm(x)[0]
        # lstm_out shape (batch_size, frames, hidden_size)

        # skip connection
        x = torch.cat([x, lstm_out], dim=-1)
        # x shape (batch_size, frames, 2*hidden_size)

        x = self.fc2(x.reshape(batch_size*frames, x.shape[-1]))
        x = self.bn2(x)
        x = torch.relu(x)
        # x shape (batch_size*frames, hidden_size)

        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        # x shape (batch_size*frames, channels*freq_bins)

        x = x.reshape(batch_size, frames, n_channels, freq_bins)
        # original shape
        x = x * mix

        return x


if __name__ == "__main__":
    
    model = UMX(frame_length=4096, frame_step=1024)
    sample = np.random.rand(5, 2, 100)
    model(sample)
    print("Hi")