import numpy as np
import itertools
from scipy import signal

def refine(complex_estimates, complex_mix, iterations=1, eps=None):
    """Function to apply the filter for some iterations
    
    parameters
    ----------
    complex_estimates: estimated complex stft of sources to refine of shape: frames, bins, channels, sources
    complex_mix: complex stft of the mixture of shape: frames, bins, channels
    iterations: iteration for the algorithm
    eps: small number to avoid division by zero

    Returns
    -------
    complex_estimates: the new refined complex estimates of the sources of shape: frames, bins, channels, sources
    """
    # use Machine limits for floating point type
    if eps is None:
        eps = np.finfo(np.real(complex_mix[0]).dtype).eps

    n_frames, n_bins, n_channels = complex_mix.shape
    n_sources = complex_estimates.shape[-1]

    # initialize the spatial covariance matrices and the PSD matrices
    # covariance between bins for all channels
    spatial_cov = np.zeros((n_bins, n_channels, n_channels, n_sources), complex_mix.dtype)
    psd = np.zeros((n_frames, n_bins, n_sources))
    regularization = np.sqrt(eps) * (np.tile(np.eye(n_channels, dtype=np.complex64), (1, n_bins, 1, 1)))    

    for i in range(iterations):

        for source_j in range(n_sources):
            # compute the spatial covariance and psd for source j
            psd[..., source_j], spatial_cov[..., source_j] = source_psd_cov(complex_estimates[..., source_j], eps)

        for frame in range(n_frames):

            # mix covariance of shape: 1 frame, bins, channels, channels
            mix_cov = mixture_cov(psd[None, frame, ...], spatial_cov) + regularization
            inv_mix_cov = invert(mix_cov, eps)
            for source_j in range(n_sources):
                complex_estimates[frame, ..., source_j] = apply_filter(
                                                            psd[None, frame, ..., source_j],
                                                            spatial_cov[..., source_j],
                                                            inv_mix_cov,
                                                            complex_mix[None, frame, ...]
                                                        )

    return complex_estimates

def wiener(mag_estimates, mix_stft, iterations=1):
    """
    Function to apply wiener filter on the estimated sources

    parameters
    ----------
    mag_estimates: Estimated Magnitude Spectrogram for each source of shape: frames, freq_bins, channels, sources
    mix_stft: complex stft of the mixture of shape: frames, freq_bins, channels
    iterations: number of iterations

    Returns
    -------
    complex_sources: Estimates of sources stft of shape: frames, bins, channels, sources
    """
    # compute the complex stft for each source A*e(j*theta) of shape: frames, freq_bins, channels, sources
    complex_sources = mag_estimates * np.exp(1j*np.angle(mix_stft[..., None]))

    if iterations == 0:
        return complex_sources
    
    max_abs = max(1, np.abs(mix_stft).max()/10.0)
    complex_sources = refine(complex_sources/max_abs, mix_stft/max_abs, iterations)
    return complex_sources*max_abs

def apply_filter(psd_j, cov_j, inv_mix_cov, mix_stft):
    """Function to apply the equation of the filter
    
    parameters
    ----------
    psd_j: psd of source j of shape: frame, bins
    cov_j: covariance of source j of shape: bins, channels, channels
    inv_mix_cov: inverse of mix covariance of shape: 1 frame, bins, channels, channels
    mix_stft: complex stft of the mixture of shape: 1 frame, bins, channels

    Returns
    -------
    estimate: complex estimate of source j of shape: 1 frame, bins, channels
    """

    n_bins, n_channels = cov_j.shape[:2]
    weight = np.zeros_like(inv_mix_cov)
    for (i1, i2, i3) in itertools.product(*(range(n_channels),)*3):
        weight[..., i1, i2] += (cov_j[None, :, i1, i3] * inv_mix_cov[..., i3, i2])
    weight *= psd_j[..., None, None]

    estimate = np.zeros((1, n_bins, n_channels), mix_stft.dtype)
    for i in range(n_channels):
        estimate += weight[..., i] * mix_stft[..., i, None]
    return estimate

def mixture_cov(sources_psd, sources_cov):
    """Function to compute mixture covariance
    
    parameters
    ----------
    sources_psd: power spectral densiries for all sources of shape: frames, bins, sources
    sources_cov: spatial covariance for all sources of shape: bins, channels, channels, sources
    
    Returns
    -------
    mix_cov: mixture covariance matrix of shape: 1 frame, bins, channels, channels
    """
    n_frames, n_bins, n_sources = sources_psd.shape
    n_channels = sources_cov.shape[1]

    mix_cov = np.zeros((n_frames, n_bins, n_channels, n_channels), sources_cov.dtype)
    for source_j in range(n_sources):
        mix_cov += sources_psd[..., source_j, None, None] * sources_cov[None, ..., source_j]
    return mix_cov

def source_psd_cov(complex_source, eps):
    """Function to compute the PSD and spatial covariance matrix for sources
    
    parameters
    ----------
    complex_source: complex stft of source of shape: n_frames, n_bins, n_channels
    eps: small value to avoid dividing by zero

    Returns
    -------
    psd, cov: psd and covariance matrices of source j
    """
    # computer average power over channels of shape: frames, bins
    psd = np.mean(np.abs(complex_source)**2, axis=2)

    n_frames, n_bins, n_channels = complex_source.shape
    cov = 0
    weight = eps
    for frame in range(n_frames):

        # covariance shape: 1 frame, bins, channels, channels
        frame_cov = np.zeros((1, n_bins, n_channels, n_channels), complex_source.dtype)
        # loop over all channels combinations
        for (i1, i2) in itertools.product(*(range(n_channels),)*2):
            frame_cov[..., i1, i2] += complex_source[None, frame, :, i1] * np.conj(complex_source[None, frame, :, i2])
        cov += frame_cov
        weight += psd[None, frame, ...]
    
    cov /= weight[..., None, None]
    return psd, cov

def invert(mix_cov, eps):
    """Function to compute the inverse of the covariance matrix
    
    parameters
    ----------
    mix_cov: mixture covariance matrix of shape: 1 frame, bins, channels, channels
    """

    n_channels = mix_cov.shape[-1]

    if n_channels == 1:
        inv = 1.0/(mix_cov+eps)
    elif n_channels == 2:
        det = mix_cov[..., 0, 0] * mix_cov[..., 1, 1] - mix_cov[..., 0, 1] * mix_cov[..., 1, 0]
        inv_det = 1.0/(det)
        inv = np.empty_like(mix_cov)
        inv[..., 0, 0] = inv_det*mix_cov[..., 1, 1]
        inv[..., 1, 1] = inv_det*mix_cov[..., 0, 0]
        inv[..., 0, 1] = -inv_det*mix_cov[..., 0, 1]
        inv[..., 1, 0] = -inv_det*mix_cov[..., 1, 0]
    else:
        print("Invalid number of channels")
    return inv


class iSTFT():

    def __init__(self, frame_length=4096, frame_step=1024):

        self.frame_length = frame_length
        self.frame_step = frame_step
        self.window = signal.get_window('hann', self.frame_length)

    def __call__(self, stft, boundary=True):
        """
        Performs the inverse Short Time Fourier Transform (iSTFT)

        Parameters
        ----------
        stft: input short time fourier transfort of type complex and shape: channels, freq, frames.
        boundary: Specifiy whether the input is extended at the boundaries.

        Returns
        -------
        audio: audio signal of shape: channels, samples.
        """

        channels, freq, frames = stft.shape
        signal_length = (frames-1) * self.frame_step + self.frame_length

        inverse = np.fft.irfft(stft, axis=-2, n=self.frame_length) * self.window.sum()
        audio = np.zeros((channels, signal_length), dtype=inverse.dtype)
        norm = np.zeros(signal_length, dtype=inverse.dtype)

        for frame in range(frames):
            
            audio[:, frame*self.frame_step : frame*self.frame_step + self.frame_length] += inverse[..., frame]*self.window
            norm[frame*self.frame_step : frame*self.frame_step + self.frame_length] += self.window**2

        if boundary:
            # Crop the extension
            audio = audio[:, self.frame_length//2:-self.frame_length//2]
            norm = norm[self.frame_length//2:-self.frame_length//2]
        
        # Normalize the audio
        audio /= np.where(norm > 1e-10, norm, 1.0)

        return audio
