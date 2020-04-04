import numpy as np
from matplotlib import pyplot as plt
import h5py
from scipy.stats import pearsonr


def eval_rate_results(ref, ests: tuple, sigs: tuple, labels: tuple):
    """
    :param ref: reference pulse rate values
    :param ests: tuple of network probability estimates
    :param sigs: tuple of intermediate signals (output of PhysNet)
    """
    assert len(ests) == len(sigs), 'Number of estimates must match number of signals!'
    n_est = len(ests)

    # Plot signal waveform
    plt.figure(figsize=(12, 8))
    w = 1000
    for i in range(n_est):
        tmp = sigs[i][:].flatten()[:w]
        tmp = (tmp-np.mean(tmp))/np.std(tmp)
        plt.plot(tmp[:w]+(n_est-i)*5, label=labels[i])
    plt.xlabel('Time [h]')
    plt.title('Estimated signal form')
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.legend()

    N = len(ref)
    Fs = 20.
    n = 128
    T = n/Fs  # length between two points in seconds
    t = np.linspace(0, N-1, N)*T/60./60.  # time vector in hours
    ref = ref*60.  # in BPM
    for i, est in enumerate(ests):
        est = est*60.
        plt.figure(figsize=(12, 6))
        plt.title(labels[i])
        plt.plot(t, ref, 'k', linewidth=2, label='reference')
        plt.plot(t, est[:, 0], 'r--', linewidth=1.5, label='mean estimate')
        plt.fill_between(t, est[:, 0] + est[:, 1], est[:, 0] - est[:, 1], color='r', alpha=0.2, label='confidence')
        plt.legend()
        plt.grid()
    plt.show()

    est_list = np.empty((ests[0].shape[0], n_est), dtype=float)
    for i, est in enumerate(ests):
        est_list[:, i] = est[:, 0]*60.    # use the expected value statistics

    # remove 6.2 to 8 hours from arrays since these parts are corrupted
    # start_rmidx = int(6.2*60*60)
    # end_rmidx = int(8*60*60)
    # idxs2rm = [x for x in range(start_rmidx, end_rmidx)]
    # ref_list = np.delete(ref, idxs2rm, axis=0)
    # est_list = np.delete(est_list, idxs2rm, axis=0)

    # Calculate metrics
    MAEs = np.mean(np.abs(np.subtract(ref, est_list)), axis=0)
    RMSEs = np.sqrt(np.mean(np.subtract(ref, est_list)**2, axis=0))
    MSEs = np.mean(np.subtract(ref, est_list)**2, axis=0)

    rs = np.empty((n_est, 1), dtype=float)
    for count in range(n_est):
        rs[count] = pearsonr(ref.squeeze(), est_list[:, count])[0]

    for i in range(n_est):
        print(f'\n({i})th statistics')
        print(f'MAE: {MAEs[i]}')
        print(f'RMSE: {RMSEs[i]}')
        print(f'MSE: {MSEs[i]}')
        print(f'Pearson r: {rs[i]}')
        print('-------------------------------------------------')


if __name__ == '__main__':
    # with h5py.File('../outputs/re_ep28.h5', 'r') as db:
    #     keys = [key for key in db.keys()]
    #     print(keys)
    #
    #     refs = db['reference'][:]
    #     ref_arr = np.empty(shape=(len(refs), 1))
    #     for i in range(len(refs)):
    #         ref_arr[i] = np.mean(refs[i])
    #
    #     print(f'Reference shape: {ref_arr.shape}')
    #     rates = db['rates'][:]
    #     print(rates.shape)
    #
    #     signal = db['signal'][:]
    #
    # with h5py.File('../outputs/re_ep93.h5') as db:
    #     rates2 = db['rates'][:]
    #     signal2 = db['signal'][:]
    #
    with h5py.File('../outputs/re_noncrop_ep93.h5') as db:
        refs = db['reference'][:]
        rates3 = db['rates'][:]
        signal3 = db['signal'][:]

    with h5py.File('../outputs/re_cnnlstm_rate.h5') as db:
        rates_rate = db['rates'][:]
        signal_rate = db['signal'][:]
    #
    # with h5py.File('../outputs/re_cnnlstm_ep35.h5') as db:
    #     rates4 = db['rates'][:]
    #     signal4 = db['signal'][:]

    # eval_rate_results(ref_arr, ests=(rates, rates2, rates3, rates4), sigs=(signal, signal2, signal3, signal4),
    #                 labels=('RateProbEst-crop-ep28', 'RateProbEst-crop-ep93', 'RateProbEst-full-ep93', 'CNN-LSTM'))
    #
    # with h5py.File('../outputs/re_cnnlstm_laplace.h5') as db:
    #     refs = db['reference'][:]
    #     rates_lap = db['rates'][:]
    #     signal_lap = db['signal'][:]
    #
    # with h5py.File('../outputs/re_cnnlstm_gauss.h5') as db:
    #     rates_gau = db['rates'][:]
    #     signal_gau = db['signal'][:]
    #
    eval_rate_results(refs, ests=(rates3, rates_rate), sigs=(signal3, signal_rate),
                      labels=('CNN', 'CNN+LSTM'))