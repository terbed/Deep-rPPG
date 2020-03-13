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
        """
        :param outputs: network output of shape: (batch_size, signal_length)
        :param targets: reference rate values: (batch_size, 1)
        """
        device = outputs.device
        if not outputs.is_cuda:
            torch.backends.mkl.is_available()

        N = outputs.shape[-1]
        pulse_band = tr.tensor([40./60., 250./60.], dtype=tr.float32)
        f = tr.linspace(0, Fs / 2, int(N / 2) + 1, dtype=tr.float32)

        min_idx = tr.argmin(tr.abs(f - pulse_band[0]))
        max_idx = tr.argmin(tr.abs(f - pulse_band[1]))

        X = tr.rfft(outputs, 1, normalized=True)
        P1 = tr.add(X[:, :, 0] ** 2, X[:, :, 1] ** 2)  # One sided Power spectral density

        # calculate indices corresponding to refs
        ref_idxs = []
        for ref in targets:
            ref_idxs.append(tr.argmin(tr.abs(f - ref)))

        # calc SNR for each batch
        losses = tr.empty((len(ref_idxs),), dtype=tr.float32)
        freq_num_in_pulse_range = max_idx - min_idx
        for count, ref_idx in enumerate(ref_idxs):
            pulse_freq_amp = P1[count, ref_idx]
            other_avrg = (tr.sum(P1[count, min_idx:ref_idx - 1]) + tr.sum(P1[count, ref_idx + 2:max_idx])) / (
                        freq_num_in_pulse_range - 3)
            losses[count] = -10 * tr.log10(pulse_freq_amp / other_avrg)

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

        mus = outputs[:, :, 0]
        bs = outputs[:, :, 1]
        s = tr.log(bs)

        losses = tr.exp(-1*s)*tr.abs(targets-mus) + s

        return losses.mean()


if __name__ == "__main__":

    crit = SNRLoss()

    outputs = tr.randn(12, 128)
    targets = tr.randn(12, 1)

    print(crit(outputs, targets))
