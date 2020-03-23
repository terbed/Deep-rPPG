"""
Loss functions for training
"""

import torch
import torch.nn as nn
tr = torch


class NegPeaLoss(nn.Module):
    def __init__(self):
        super(NegPeaLoss, self).__init__()

    def forward(self, x, y):
        if len(x.size()) == 1:
            x = tr.unsqueeze(x, 0)
            y = tr.unsqueeze(y, 0)
        T = x.shape[1]
        p_coeff = tr.sub(T * tr.sum(tr.mul(x, y), 1), tr.mul(tr.sum(x, 1), tr.sum(y, 1)))
        norm = tr.sqrt((T * tr.sum(x ** 2, 1) - tr.sum(x, 1) ** 2) * (T * tr.sum(y ** 2, 1) - tr.sum(y, 1) ** 2))
        p_coeff = tr.div(p_coeff, norm)
        losses = tr.tensor(1.) - p_coeff
        totloss = tr.mean(losses)
        return totloss


class SNRLoss(nn.Module):
    def __init__(self):
        super(SNRLoss, self).__init__()

    def forward(self, outputs: tr.Tensor, targets: tr.Tensor, Fs=20):
        device = outputs.device
        if not outputs.is_cuda:
            torch.backends.mkl.is_available()

        N = outputs.shape[-1]
        pulse_band = tr.tensor([40/60., 250/60.], dtype=tr.float32).to(device)
        f = tr.linspace(0, Fs/2, int(N/2)+1, dtype=tr.float32).to(device)

        min_idx = tr.argmin(tr.abs(f - pulse_band[0]))
        max_idx = tr.argmin(tr.abs(f - pulse_band[1]))

        outputs = outputs.view(-1, N)
        targets = targets.view(-1, 1)

        X = tr.rfft(outputs, 1, normalized=True)
        P1 = tr.add(X[:, :, 0]**2, X[:, :, 1]**2)                                   # One sided Power spectral density

        # calculate indices corresponding to refs
        ref_idxs = []
        for ref in targets:
            ref_idxs.append(tr.argmin(tr.abs(f-ref)))

        # calc SNR for each batch
        losses = tr.empty((len(ref_idxs),), dtype=tr.float32)
        freq_num_in_pulse_range = max_idx-min_idx
        for count, ref_idx in enumerate(ref_idxs):
            pulse_freq_amp = P1[count, ref_idx]
            other_avrg = (tr.sum(P1[count, min_idx:ref_idx-1]) + tr.sum(P1[count, ref_idx+2:max_idx]))/(freq_num_in_pulse_range-3)
            losses[count] = -10*tr.log10(pulse_freq_amp/other_avrg)

        return tr.mean(losses)


class _SNRLoss(nn.Module):
    def __init__(self):
        super(_SNRLoss, self).__init__()

    def forward(self, outputs: tr.Tensor, targets: tr.Tensor, Fs=20):
        """
        :param outputs: network output of shape: (batch_size, signal_length)
        :param targets: reference rate values: (batch_size, 1)
        """
        device = outputs.device
        if not outputs.is_cuda:
            torch.backends.mkl.is_available()

        N = outputs.shape[-1]
        targets = targets.view(-1, 1)
        outputs = outputs.view(-1, N)
        # print(outputs.shape, targets.shape)

        pulse_band = tr.tensor([40./60., 250./60.], dtype=tr.float32).to(device)
        f = tr.linspace(0, Fs / 2, int(N / 2) + 1, dtype=tr.float32).to(device)

        min_idx = tr.argmin(tr.abs(f - pulse_band[0]))
        max_idx = tr.argmin(tr.abs(f - pulse_band[1]))

        X = tr.rfft(outputs, 1, normalized=True)
        # print(f'X.shape: {X.shape}')
        P1 = tr.add(X[:, :, 0] ** 2, X[:, :, 1] ** 2)  # One sided Power spectral density

        # calculate indices corresponding to refs
        ref_idxs = []
        for ref in targets:
            ref_idxs.append(tr.argmin(tr.abs(f - ref)))

        # calc SNR for each batch
        losses = tr.empty((len(ref_idxs),), dtype=tr.float32)
        for count, ref_idx in enumerate(ref_idxs):
            # Creating template for second and first harmonics pulse energy
            u = tr.zeros(len(f))
            u[ref_idx] = u[ref_idx*2] = 1
            u[ref_idx-1] = u[ref_idx+1] = 0.5
            # u[ref_idx*2 - 1] = u[ref_idx*2 + 1] = 0.8
            u = u.to(device)

            # Creating template for noise energy in pulse band
            w = tr.tensor(1) - u
            w[0:min_idx] = w[max_idx:] = 0
            w[ref_idx * 2 - 1] = w[ref_idx * 2 + 1] = 0     # do not penalize second harmonics
            w = w.to(device)

            signal_energy = tr.sum(tr.mul(P1[count, :], u))
            noise_energy = tr.sum(tr.mul(P1[count, :], w))

            losses[count] = -10 * tr.log10(signal_energy / (noise_energy+1))

        return tr.mean(losses)


class GaussLoss(nn.Module):
    """
    Loss for normal distribution (L2 like loss function)
    """

    def __init__(self):
        super().__init__()

    def forward(self, outputs: tr.Tensor, targets: tr.Tensor) -> tr.Tensor:
        """
        :param outputs: tensor of shape: (batch_num, samples_num, density_parameters), density_parameters=2 -> mu, sigma
        :param targets: tensor of shape: (batch_num, samples_num)
        :return: loss (scalar)
        """

        n_samples = targets.shape[-1]
        outputs = outputs.view(-1, n_samples, 2)
        targets = targets.view(-1, n_samples)

        mus = outputs[:, :, 0]
        sigmas = outputs[:, :, 1]
        s = tr.log(sigmas**2)

        losses = tr.exp(-1*s)*(targets-mus)**2 + s

        return losses.mean()


class LaplaceLoss(nn.Module):
    """
    Loss for Laplace distribution (L1 like loss function)
    """

    def __init__(self):
        super().__init__()

    def forward(self, outputs: tr.Tensor, targets: tr.Tensor) -> tr.Tensor:
        """
        :param outputs: tensor of shape: (batch_num, samples_num, density_parameters), density_parameters=2 -> mu, b
        :param targets: tensor of shape: (batch_num, samples_num)
        :return: loss (scalar)
        """

        n = targets.shape[-1]
        outputs = outputs.view(-1, n, 2)
        targets = targets.view(-1, n)

        mus = outputs[:, :, 0]
        bs = outputs[:, :, 1]
        s = tr.log(bs)

        losses = tr.exp(-1*s)*tr.abs(targets-mus) + s

        return losses.mean()


if __name__ == "__main__":

    crit = SNRLoss()

    outputs = tr.randn(128)
    targets = tr.randn(1)

    print(crit(outputs, targets))
