
import numpy as np
import scipy.signal as signal
from scipy.stats import zscore

def lowpass_filter(data, fs, cutoff=1.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def bandpass_filter(data, fs, lowcut=0.5, highcut=2.5, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

def normalize_signal(data):
    """
    Apply Z-score normalization to the signal.
    """
    return zscore(data)

def segment_signal(data, window_size, step_size):
    """
    Segment 1D signal into overlapping windows.
    Args:
        data: 1D array of signal data
        window_size: Number of samples in a window
        step_size: Number of samples to shift
    Returns:
        segments: 2D array of shape (num_segments, window_size)
    """
    segments = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        segments.append(data[start:end])
    return np.array(segments)
