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


def eval_ppg(ref, est, Fs=20, pulse_band=(80./60., 250./60.)):
    """
    Calculates statistics

    :param ref: reference numerical HR signal
    :param est: the estimated ppg signal
    :param Fs: sampling rate
    :param pulse_band: valid pulse range
    """

    stride = Fs     # 1 sec stride
    w = 512         # ~30 sec window (if Fs=20)
    hamming = np.hamming(w)

    # Calculating filter parameters
    b, a = butter_bandpass(pulse_band[0], pulse_band[1], Fs, order=3)

    # Frequency vector of the window
    f = np.linspace(0, Fs/2, w//2-1)

    # Find indices corresponding to pulse boundaries
    min_pulse_idx = np.argmin(np.abs(f - pulse_band[0]))
    max_pulse_idx = np.argmin(np.abs(f - pulse_band[1]))

    ref_list = []
    est_list = []
    n_segm = len(est - w) // stride

    for i in range(n_segm-1):
        pass


