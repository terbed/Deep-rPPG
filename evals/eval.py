import numpy as np
from matplotlib import pyplot as plt
import h5py
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr, mode
from tqdm import tqdm


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def eval_signal_results(ref, ests: tuple, Fs=20, pulse_band=(50., 250.), is_plot=False):
    """
    Calculates statistics

    :param ref: reference numerical HR signal
    :param ests: the estimated ppg signal
    :param Fs: sampling rate
    :param pulse_band: valid pulse range
    """
    N = len(ests[-1].flatten())
    L = (N-1)/Fs
    t = np.linspace(0, L, N)/60/60

    stride = Fs     # 1 sec stride
    w = 512         # ~30 sec window (if Fs=20)
    hanning = np.hanning(w)

    # Calculating filter parameters
    b, a = butter_bandpass(pulse_band[0]/60., pulse_band[1]/60., Fs, order=3)

    # Frequency vector of the window
    f = np.linspace(0, Fs/2, w//2+1)*60.

    # Find indices corresponding to pulse boundaries
    min_pulse_idx = np.argmin(np.abs(f - pulse_band[0]))
    max_pulse_idx = np.argmin(np.abs(f - pulse_band[1]))

    n_segm = (len(ests[-1])-w) // stride
    n_est = len(ests)
    # n_segm = 1000
    ref_list = np.empty((1, n_segm), dtype=float)
    est_list = np.empty((n_est, n_segm), dtype=float)
    SNR_list = np.empty((n_est, n_segm), dtype=float)

    for i in tqdm(range(n_segm)):
        # Select the most frequent pulse value as reference
        curr_ref = mode(ref[i*stride:i*stride+w])[0]
        ref_idx = np.argmin(np.abs(f-curr_ref))
        ref_list[0, i] = curr_ref

        # Iterate through estimates
        for count, est in enumerate(ests):
            # Calculate estimated signal maximum component and SNR
            est = est.flatten()
            curr_res_seg = est[i * stride:i * stride + w]
            # filter
            curr_res_seg = filtfilt(b, a, curr_res_seg)
            # windowing to eliminate leakage
            curr_res_seg = np.multiply(curr_res_seg, hanning)
            # calculate fft
            freq_dom = np.fft.rfft(curr_res_seg)/len(curr_res_seg)
            power_spect = np.abs(np.multiply(freq_dom, freq_dom.conj()))

            max_idx = min_pulse_idx + np.argmax(power_spect[min_pulse_idx:max_pulse_idx])
            est_pr = f[max_idx]
            est_list[count, i] = est_pr

            # Calculating SNR
            u = np.zeros(len(power_spect))
            u[ref_idx-2:ref_idx+3] = u[ref_idx*2-2:ref_idx*2+3] = 1

            signal_energy = np.sum(np.multiply(power_spect, u))
            noise_energy = np.sum(np.multiply(power_spect[min_pulse_idx:max_pulse_idx], (1-u)[min_pulse_idx:max_pulse_idx]))

            snr = 10*np.log10(signal_energy/noise_energy)
            SNR_list[count, i] = snr

    # Save results
    ests_arr = np.empty((n_est, N))
    for i, e in enumerate(ests):
        ests_arr[i, :] = e.flatten()[:N]
    with h5py.File('eval_data.h5', 'w') as db:
        db.create_dataset('ref_orig', shape=ref.shape, dtype=np.float32, data=ref)
        db.create_dataset('ests', shape=ests_arr.shape, dtype=np.float32, data=ests_arr)
        db.create_dataset('ref_list', shape=ref_list.shape, dtype=np.float32, data=ref_list)
        db.create_dataset('est_list', shape=est_list.shape, dtype=np.float32, data=est_list)
        db.create_dataset('SNR_list', shape=SNR_list.shape, dtype=np.float32, data=SNR_list)

    # Calculate statistics
    MAEs = np.mean(np.abs(np.subtract(ref_list, est_list)), axis=1)
    RMSEs = np.sqrt(np.mean(np.subtract(ref_list, est_list)**2, axis=1))
    MSEs = np.mean(np.subtract(ref_list, est_list)**2, axis=1)
    MSNRs = np.mean(SNR_list, axis=1)

    rs = np.empty((n_est, 1), dtype=float)
    for count, est in enumerate(est_list):
        rs[count] = pearsonr(ref_list.squeeze(), est)[0]

    for i in range(n_est):
        print(f'\n({i})th statistics')
        print(f'MAE: {MAEs[i]}')
        print(f'RMSE: {RMSEs[i]}')
        print(f'MSE: {MSEs[i]}')
        print(f'Pearson r: {rs[i]}')
        print(f'MSNR: {MSNRs[i]}')
        print('-------------------------------------------------')

    # -------------------------------
    # Visualizations
    # -------------------------------
    if is_plot:
        # Plot signal waveform
        plt.figure(figsize=(12, 6))
        for i in range(n_est):
            tmp = ests[i].flatten()[:w]
            plt.plot(t[:w], tmp[:w], label=f'{i}th result')
        plt.xlabel('Time [h]')
        plt.title('Estimated signal form')
        plt.legend()
        plt.show()

        # Plot pulse rates
        tt = [x/60./60. for x in range(est_list.shape[-1])]
        plt.figure(figsize=(12, 6))
        plt.plot(tt, ref_list[0, :len(tt)], color='k', label='reference')
        for i in range(n_est):
            plt.plot(tt, est_list[i], label=f'{i}th result')
        plt.grid()
        plt.xlabel('Time [h]')
        plt.ylabel('PR [BPM]')
        plt.title('Estimated and reference PR values by different models')
        plt.legend(loc='best')
        plt.show()


def eval_rate_results(ref, ests: tuple):
    n_est = len(ests)

    N = len(ref)
    Fs = 20.
    n = 128
    T = n/Fs  # length between two points in seconds
    t = np.linspace(0, N-1, N)*T/60./60.  # time vector in hours
    ref = ref*60.  # in BPM
    for i, est in enumerate(ests):
        est = est*60.
        plt.figure(figsize=(12, 6))
        plt.title(f'RateProbEst network result: {i}')
        plt.plot(t, ref, 'k', linewidth=2, label='reference')
        plt.plot(t, est[:, 0], 'r--', linewidth=1.5, label='mean estimate')
        plt.fill_between(t, est[:, 0] + est[:, 1], est[:, 0] - est[:, 1], color='r', alpha=0.2, label='confidence')
        plt.legend()
        plt.grid()
    plt.show()

    est_list = np.empty((ests[0].shape[0], n_est), dtype=float)
    for i, est in enumerate(ests):
        est_list[:, i] = est[:, 0]*60.  # use the expected value statistics

    # Calculate metrics
    MAEs = np.mean(np.abs(np.subtract(ref, est_list)), axis=0)
    RMSEs = np.sqrt(np.mean(np.subtract(ref, est_list)**2, axis=0))
    MSEs = np.mean(np.subtract(ref, est_list)**2, axis=0)

    rs = np.empty((n_est, 1), dtype=float)
    for count in range(n_est):
        rs[count] = pearsonr(ref_list.squeeze(), est_list[:, count])[0]

    for i in range(n_est):
        print(f'\n({i})th statistics')
        print(f'MAE: {MAEs[i]}')
        print(f'RMSE: {RMSEs[i]}')
        print(f'MSE: {MSEs[i]}')
        print(f'Pearson r: {rs[i]}')
        print('-------------------------------------------------')


def eval_signal_results_from_h5(path):
    with h5py.File(path, 'r') as db:
        ref_list = db['ref_list'][:]
        est_list = db['est_list'][:]
        ests = db['ests'][:]
        SNR_list = db['SNR_list'][:]

    Fs = 20
    w = 512
    n_est = len(ests)
    N = len(ests[-1].flatten())
    L = (N-1)/Fs
    t = np.linspace(0, L, N)/60/60

    # Calculate statistics
    MAEs = np.mean(np.abs(np.subtract(ref_list, est_list)), axis=1)
    RMSEs = np.sqrt(np.mean(np.subtract(ref_list, est_list)**2, axis=1))
    MSEs = np.mean(np.subtract(ref_list, est_list)**2, axis=1)
    MSNRs = np.mean(SNR_list, axis=1)

    rs = np.empty((n_est, 1), dtype=float)
    for count, est in enumerate(est_list):
        rs[count] = pearsonr(ref_list.squeeze(), est)[0]

    for i in range(n_est):
        print(f'\n({i})th statistics')
        print(f'MAE: {MAEs[i]}')
        print(f'RMSE: {RMSEs[i]}')
        print(f'MSE: {MSEs[i]}')
        print(f'Pearson r: {rs[i]}')
        print(f'MSNR: {MSNRs[i]}')
        print('-------------------------------------------------')

    # -------------------------------
    # Visualizations
    # -------------------------------
    # Plot signal waveform
    plt.figure(figsize=(12, 8))
    shift = 0
    w = 1000
    for i in range(n_est):
        tmp = ests[i, :].flatten()[:w]
        tmp = (tmp-np.mean(tmp))/np.std(tmp)
        plt.plot(t[:w], tmp[:w]+(n_est-i)*5, label=f'{i}th result')
    plt.xlabel('Time [h]')
    plt.title('Estimated signal form')
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.legend()

    # Plot pulse rates
    tt = [x/60./60. for x in range(est_list.shape[-1])]
    n = len(tt)

    n_plot = 3
    fig, axs = plt.subplots(n_plot, 1, figsize=(12, 8))
    for i, ax in enumerate(axs):
        ax.plot(tt[:n//2], ref_list[0, :n//2], color='k', label='reference', linewidth=2.)
        ax.plot(tt[:n//2], est_list[n_est-i-1][:n//2], 'r--', label=f'{i}th result', alpha=0.8)
        ax.grid()
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('PR [BPM]')
        ax.set_ylim(80, 250)
        ax.legend(loc='upper right')
    fig.subplots_adjust(hspace=1)
    fig.subplots_adjust(top=0.99, bottom=0.05, left=0.05, right=0.95)

    fig, axs = plt.subplots(n_plot, 1, figsize=(12, 8))
    for i, ax in enumerate(axs):
        ax.plot(tt[n//2:], ref_list[0, n//2:], color='k', label='reference', linewidth=2.)
        ax.plot(tt[n//2:], est_list[n_est-i-1][n//2:], 'r--', label=f'{i}th result', alpha=0.8)
        ax.grid()
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('PR [BPM]')
        ax.set_ylim(80, 250)
        ax.legend(loc='upper right')
    fig.subplots_adjust(hspace=1)
    fig.subplots_adjust(top=0.99, bottom=0.05, left=0.05, right=0.95)
    plt.show()


if __name__ == '__main__':

    signal = False
    calc = False

    if signal:
        if calc:
            ref = np.loadtxt('../outputs/benchmark_minden_reference.dat')

            # est1 = np.loadtxt('../outputs/dp190111-benchmark_minden.dat')
            # est2 = np.loadtxt('../outputs/dp200101-benchmark_minden.dat')
            #
            # est3 = np.loadtxt('../outputs/pn190111-benchmark_minden.dat')
            # est4 = np.loadtxt('../outputs/pn190111_imgaugm-benchmark_minden.dat')
            # est5 = np.loadtxt('../outputs/pn190111_allaugm-benchmark_minden.dat')

            est6 = np.loadtxt('../outputs/PhysNet-tPIC191111_SNRLoss-onLargeBenchmark-200301-res.dat')
            est7 = np.loadtxt('../outputs/pn191111snr_imgaugm-benchmark_minden.dat')
            est8 = np.loadtxt('../outputs/pn191111snr_allaugm-benchmark_minden.dat')

            eval_signal_results(ref, (est6, est7, est8))
        else:
            # eval_results_from_h5('eval_data.h5')

            est = np.loadtxt('../outputs/re_test.dat')
            eval_rate_results(est[:, 0], (est,))
    else:
        with h5py.File('../outputs/re_ep28.h5', 'r') as db:
            keys = [key for key in db.keys()]
            print(keys)

            refs = db['reference'][:]
            ref_list = np.empty(shape=(len(refs), 1))
            for i in range(len(refs)):
                ref_list[i] = mode(refs[i, :])[0]

            print(ref_list.shape)
            rates = db['rates'][:]
            print(rates.shape)
            signal = db['signal'][:]

        with h5py.File('../outputs/re_ep93.h5') as db:
            rates2 = db['rates'][:]
            signal2 = db['signal'][:]

        plt.figure()
        plt.title('output of the first network')
        plt.plot(signal, label='ep28')
        plt.plot(signal2 + 4*np.std(signal2), label='ep93')
        plt.show()
        eval_rate_results(ref_list, (rates, rates2))


