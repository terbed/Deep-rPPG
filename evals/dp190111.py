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

# ref_path = '/media/nas/PUBLIC/0_training_set/pulse/PPGbenchmark_visible_128x128_8UC3_box.hdf5'
ref_path = '/Volumes/sztaki/DATA/PPGbenchmark_visible_128x128_8UC3_box.hdf5'
out_path = '../outputs/dp190111-PPGbenchmark_visible.dat'

output = np.loadtxt(out_path)
print(output.shape)

with h5py.File(ref_path, 'r') as db:
    ref = db['references']
    print([key for key in ref.keys()])
    ppg_ref = ref['PPGSignal'][:]
    pr_ref = ref['PulseNumerical'][:]

out_filt = filtfilt(b, a, output)
n = len(out_filt)
t = np.arange(n)/20/60
print(n, ppg_ref.shape)

plt.figure(figsize=(12, 6))
plt.specgram(out_filt, NFFT=512, Fs=20*60, noverlap=492, vmin=-32)
plt.plot(t, pr_ref[:n], label='ref')
plt.ylim(80, 250)
plt.ylabel('Pulse [BPM]')
plt.ylabel('Time [min]')
plt.colorbar()
plt.legend()


plt.figure(figsize=(12, 6))
plt.plot(t, ppg_ref[:n], label='ref')
plt.plot(t, out_filt, label='out')
plt.grid()
plt.ylabel('Time [min]')
plt.legend()
plt.show()


pr = pearsonr(out_filt, ppg_ref[:n])[0]
print(f'\nPearson correlation coeff: {pr}')
