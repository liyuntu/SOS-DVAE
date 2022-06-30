import numpy as np
from numpy import logical_and
from scipy.io import loadmat
from scipy.signal import welch
from scipy.integrate import simps
import pandas as pd


# Function calculates power spectral densities for a signal
def calculate_psd(signal, samp_freq, freq_bands):
    """
    Description
    -----------
    Given a time series signal, signal sampling frequency, and a numpy array of frequency bands
    (number of bands x [frequency low, frequency high]), returns a 1-dimensional numpy array of PSD values.

    Parameters
    ----------
    signal : numpy.ndarray; numpy array of signal
    samp_freq :  float; signal sampling frequency
    freq_bands :  numpy.ndarray; 2-dimensional (number of bands x [frequency low, frequency high]) numpy array of
                  frequency bands

    Returns
    -------
    psd_vals :  numpy.ndarray; 1-dimensional numpy array of PSD values for a given signal
    """

    # Initialize PSD values list
    psd_vals = []

    # Iterate through frequency bands
    for i in range(len(freq_bands)):
        # Extract single frequency band
        freq_band = freq_bands[i, :]

        # Define window length
        freqs, psd = welch(x=signal, fs=samp_freq, nperseg=len(signal))
        band_idx = logical_and(freq_band[0] <= freqs, freqs <= freq_band[1])
        freq_res = abs(freqs[1] - freqs[0])
        power = simps(psd[band_idx], dx=freq_res)
        psd_vals.append(power)

    # Return PSD values
    return psd_vals


# PSD features function
def get_psd_features(eeg_mat, samp_freq, freq_bands):
    """
    Description
    -----------
    Given a matrix of EEG signals (channels x time), sampling frequency, and a numpy array of frequency bands
    (number of bands x [frequency low, frequency high]), returns a 1-dimensional array of PSD features.

    Parameters
    ----------
    eeg_mat :  numpy.ndarray; 2-dimensional (channels x time) numpy array of EEG signals for a single observation
    samp_freq :  float; EEG sampling frequency
    freq_bands :  numpy.ndarray; 2-dimensional (number of bands x [frequency low, frequency high]) numpy array of
                  frequency bands

    Returns
    -------
    psd_feat :  numpy.ndarray; 1-dimensional numpy array of PSD features
    """

    # Initialize features list
    psd_feat_list = []

    # Iterate through electrode channels
    for channel in range(len(eeg_mat)):
        # Get PSD features for channel EEG signal
        eeg_signal = eeg_mat[channel, :]
        psd_feat_list.append(calculate_psd(signal=eeg_signal, samp_freq=samp_freq,
                                           freq_bands=freq_bands))

    # Convert list of PSD features to numpy array
    psd_feat = np.asarray(psd_feat_list).ravel()

    # Return subject PSD features
    return psd_feat

