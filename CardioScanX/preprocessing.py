from scipy.signal import butter, filtfilt, iirnotch
import numpy as np


def bandpass_filter(sig, lowcut=0.5, highcut=45.0, fs=500.0, order=4):
    # sig: numpy array [12, T]
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    out = filtfilt(b, a, sig, axis=1)
    return out


def notch_filter(sig, freq=50.0, fs=500.0, quality=30.0):
    b, a = iirnotch(freq / (fs/2.0), quality)
    out = filtfilt(b, a, sig, axis=1)
    return out


def normalize(sig):
    # zero mean per-lead and scale to unit variance
    mean = np.mean(sig, axis=1, keepdims=True)
    std = np.std(sig, axis=1, keepdims=True) + 1e-8
    return (sig - mean) / std


def apply_preprocessing(sig, bandpass=False, notch=False, normalize_sig=False):
    out = sig.copy()
    if bandpass:
        out = bandpass_filter(out)
    if notch:
        out = notch_filter(out)
    if normalize_sig:
        out = normalize(out)
    return out
