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


Fs = 20
pulse_band = [80./60., 250./60.]
b, a = butter_bandpass(pulse_band[0], pulse_band[1], Fs, order=3)

ref_path = '/media/nas/PUBLIC/0_training_set/pulse/PPGbenchmark_visible_128x128_8UC3_box.hdf5'
out_path = '../outputs/dp190111-PPGbenchmark_visible.dat'

output = np.loadtxt(out_path)
print(output.shape)

with h5py.File(ref_path, 'r') as db:
    ref = db['references']
    ppg_ref = ref['PPGSignal'][:]

out_filt = filtfilt(b, a, output)
plt.figure(figsize=(6, 12))
plt.plot(ppg_ref, label='ref')
plt.plot(out_filt, label='out')
plt.grid()
plt.legend()
plt.show()

w = 128
n = len(out_filt)
segnum = n // w

pear_list = []
for i in range(segnum-1):
    pear_list.append(pearsonr(out_filt[i*w:(i+1)*w], ppg_ref[i*w:(i+1)*w])[0])

print(f'\nPearson correlation coeff: {np.mean(pear_list)}')
