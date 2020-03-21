from src.errfuncs import *
import torch
import numpy as np
import math
tr = torch


def snrloss_test():
    device = tr.device('cuda') if tr.cuda.is_available() else tr.device('cpu')
    n = 128
    Fs = 20.
    t = tr.linspace(0, n - 1, n, dtype=tr.float32) / Fs     # Time vector in minutes
    crit = SNRLoss()

    def create_signal(pr, noise_std):
        amp = 1.

        x = tr.tensor(amp) * tr.sin(2 * math.pi * pr * t)
        noise = tr.from_numpy(np.random.normal(0, noise_std, n))
        x = x + noise

        return x

    pr_ref = tr.tensor(150. / 60.)
    pr_sig = tr.tensor(300. / 60.)
    signal = create_signal(pr_sig, 0.5)
    print(crit(signal, pr_ref))


if __name__ == '__main__':
    snrloss_test()
