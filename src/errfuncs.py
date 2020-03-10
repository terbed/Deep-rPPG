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
        losses = 1 - p_coeff
        totloss = tr.mean(losses)
        return totloss


class SNRLoss(nn.Module):
    def __init__(self):
        super(SNRLoss, self).__init__()

    def forward(self, output: tr.Tensor, target: tr.Tensor, Fs=20):
        device = output.device
        if not output.is_cuda:
            torch.backends.mkl.is_available()

        N = output.shape[-1]
        pulse_band = tr.tensor([40, 250], dtype=tr.float32).to(device)
        f = tr.linspace(0, Fs / 2, int(N / 2) + 1, dtype=tr.float32).to(device)

        min_idx = tr.argmin(tr.abs(f - pulse_band[0] / 60))
        max_idx = tr.argmin(tr.abs(f - pulse_band[1] / 60))

        # If there is no batch extend with fantom batch dimension
        if len(output.size()) == 1:
            output = output.unsqueeze(dim=0)
            target = target.unsqueeze(dim=0)

        X = tr.rfft(output, 1, normalized=True)
        P1 = tr.add(X[:, :, 0] ** 2, X[:, :, 1] ** 2)  # One sided Power spectral density

        # Select unique reference pulse rates in current time interval
        refs = tr.mode(target)[0]

        # calculate indices corresponding to refs
        ref_idxs = []
        for ref in refs:
            ref_idxs.append(tr.argmin(tr.abs(f - ref / 60)))

        # calc SNR for each batch
        losses = tr.empty((len(ref_idxs),), dtype=tr.float32)
        freq_num_in_pulse_range = max_idx - min_idx
        for count, ref_idx in enumerate(ref_idxs):
            pulse_freq_amp = P1[count, ref_idx]
            other_avrg = (tr.sum(P1[count, min_idx:ref_idx - 1]) + tr.sum(P1[count, ref_idx + 2:max_idx])) / (
                        freq_num_in_pulse_range - 3)
            losses[count] = -10 * tr.log10(pulse_freq_amp / other_avrg)

        return tr.mean(losses)


class NLPLoss(tr.nn.Module):
    """
    Negative Log Probability loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, targets: tr.Tensor, outputs: tr.Tensor) -> tr.Tensor:
        """
        :param targets: tensor of shape: (batch_num, samples_num)
        :param outputs: tensor of shape: (batch_num, samples_num, density_parameters), density_parameters=2 -> mu, sigma
        :return: loss (scalar)
        """

        batch_num = targets.shape[0]
        mus = outputs[:, :, 0].reshape(batch_num, -1)
        sigmas = outputs[:, :, 1].reshape(batch_num, -1)

        normal_dist = tr.distributions.normal.Normal(mus, sigmas)
        losses = -1 * normal_dist.log_prob(targets)

        return losses[~torch.isnan(losses)].mean()
