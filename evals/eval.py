import numpy as np
from matplotlib import pyplot as plt
import h5py
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def eval_ppg(ref, est, Fs, pulse_band):
    """
    Calculates statistics

    :param ref: reference numerical HR signal
    :param est: the estimated ppg signal
    """

    stride = Fs     # 1 sec stride
    w = 512         # ~30 sec window (if Fs=20)


