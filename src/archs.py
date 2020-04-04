"""
This module contains the used architectures for pulse signal and pulse rate extracton
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
tr = torch


# -------------------------------------------------------------------------------------------------------------------
# PhysNet network
# -------------------------------------------------------------------------------------------------------------------
class PhysNetED(nn.Module):
    def __init__(self):
        super().__init__()

        self.start = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )

        # 1x
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # encoder
        self.encoder = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        #
        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # decoder to reach back initial temporal length
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        self.end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )

    def forward(self, x):
        x = self.start(x)
        x = self.loop1(x)
        x = self.encoder(x)
        x = self.loop4(x)
        x = self.decoder(x)
        x = self.end(x)

        return x


# -------------------------------------------------------------------------------------------------------------------
# DeepPhys network
# -------------------------------------------------------------------------------------------------------------------
class DeepPhys(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask1 = None
        self.mask2 = None

        # Appearance stream
        self.a_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.a_bn1 = nn.BatchNorm2d(32)

        self.a_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.a_bn2 = nn.BatchNorm2d(32)
        self.a_d1 = nn.Dropout2d(p=0.50)

        self.a_softconv1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.a_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.a_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.a_bn3 = nn.BatchNorm2d(64)

        self.a_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.a_bn4 = nn.BatchNorm2d(64)
        self.a_d2 = nn.Dropout2d(p=0.50)
        self.a_softconv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Motion stream
        self.m_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.m_bn1 = nn.BatchNorm2d(32)
        self.m_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.m_bn2 = nn.BatchNorm2d(32)
        self.d1 = nn.Dropout2d(p=0.50)

        self.m_avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.m_bn3 = nn.BatchNorm2d(64)
        self.m_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.m_bn4 = nn.BatchNorm2d(64)
        self.d2 = nn.Dropout2d(p=0.50)
        self.m_avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Fully conected blocks
        self.d3 = nn.Dropout(p=0.25)
        self.fully1 = nn.Linear(in_features=64 * 9 * 9, out_features=128, bias=True)
        self.fully2 = nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, A, M):
        # (A) - Appearance stream -------------------------------------------------------------
        # First two convolution layer
        A = tr.tanh(self.a_bn1(self.a_conv1(A)))
        A = tr.tanh(self.a_bn2(self.a_conv2(A)))
        A = self.a_d1(A)

        # Calculating attention mask1 with softconv1
        mask1 = tr.sigmoid(self.a_softconv1(A))
        B, _, H, W = A.shape
        norm = 2 * tr.norm(mask1, p=1, dim=(1, 2, 3))
        norm = norm.reshape(B, 1, 1, 1)
        mask1 = tr.div(mask1 * H * W, norm)
        self.mask1 = mask1

        # Pooling
        A = self.a_avgpool(A)
        # Last two convolution
        A = tr.tanh(self.a_bn3(self.a_conv3(A)))
        A = tr.tanh(self.a_bn4(self.a_conv4(A)))
        A = self.a_d2(A)

        # Calculating attention mask2 with softconv2
        mask2 = tr.sigmoid(self.a_softconv2(A))
        B, _, H, W = A.shape
        norm = 2 * tr.norm(mask2, p=1, dim=(1, 2, 3))
        norm = norm.reshape(B, 1, 1, 1)
        mask2 = tr.div(mask2 * H * W, norm)
        self.mask2 = mask2

        # (M) - Motion stream --------------------------------------------------------------------
        M = tr.tanh(self.m_bn1(self.m_conv1(M)))
        M = self.m_bn2(self.m_conv2(M))
        M = tr.tanh(tr.mul(M, mask1))  # multiplying with attention mask1
        M = self.d1(M)  # Dropout layer 1
        # Pooling
        M = self.m_avgpool1(M)
        # Last convs
        M = tr.tanh(self.m_bn3(self.m_conv3(M)))
        M = self.m_bn4(self.m_conv4(M))
        M = tr.tanh(tr.mul(M, mask2))  # multiplying with attention mask2
        M = self.d2(M)  # Dropout layer 2
        M = self.m_avgpool2(M)

        # (F) - Fully connected part -------------------------------------------------------------
        # Flatten layer out
        out = tr.flatten(M, start_dim=1)  # start_dim=1 to handle batches
        out = self.d3(out)  # dropout layer 3
        out = tr.tanh(self.fully1(out))
        out = self.fully2(out)

        return out


# -------------------------------------------------------------------------------------------------------------------
# Rate estimator probability network  TODO: Build from config file -> hyperparameter optimization
# -------------------------------------------------------------------------------------------------------------------
class RateProbEst(nn.Module):
    def __init__(self):
        super().__init__()

        max_pool_kernel_size = 5
        conv_kernel_size = 17
        padn = 8

        self.first_part = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Dropout(0.1),

            nn.Conv1d(1, 32, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(32),
            nn.ELU(),

            nn.Dropout(0.1),
            nn.Conv1d(32, 32, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(32),
            nn.ELU(),

            nn.Dropout(0.1),
            nn.Conv1d(32, 64, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU()
        )

        input_count = 64
        output_count = input_count
        self.middle_part = nn.Sequential(
            nn.Dropout(0.15),
            nn.Conv1d(64, 128, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.Dropout(0.15),
            nn.Conv1d(128, 256, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.Dropout(0.2),
            nn.Conv1d(256, 128, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.Dropout(0.3),
            nn.Conv1d(128, 64, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(64),
            nn.ELU(),

            nn.Dropout(0.3),
            nn.Conv1d(64, 32, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU()
        )

        self.end_part = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool1d(output_size=2)
        )

    def forward(self, x):
        x = x - torch.mean(x, dim=-1).unsqueeze(dim=-1)
        x = self.first_part(x)
        x = self.middle_part(x)
        x = self.end_part(x)

        x[:, :, 1] = F.elu(x[:, :, 1]) + 1.  # sigmas must have positive!

        # output shape [1, 1, 2]
        return x


# -------------------------------------------------------------------------------------------------------------------
# Rate estimator (simple point estimate)
# -------------------------------------------------------------------------------------------------------------------
class RateEst(nn.Module):
    def __init__(self):
        super().__init__()

        avg_pool_kernel_size = 5
        conv_kernel_size = 17
        padn = 8

        self.first_part = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Dropout(0.1),

            nn.Conv1d(1, 32, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(32),
            nn.ELU(),

            nn.Dropout(0.1),
            nn.Conv1d(32, 64, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(64),
            nn.ELU(),

            nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.AvgPool1d(kernel_size=avg_pool_kernel_size, stride=2),
            nn.BatchNorm1d(128),
            nn.ELU()
        )

        input_count = 128
        output_count = input_count
        self.middle_part = nn.Sequential(
            nn.Dropout(0.15),
            nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(output_count),
            nn.ELU(),

            nn.Dropout(0.15),
            nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(output_count),
            nn.ELU(),

            nn.Dropout(0.2),
            nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(output_count),
            nn.ELU(),

            nn.Dropout(0.3),
            nn.Conv1d(128, 64, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(64),
            nn.ELU(),

            nn.Dropout(0.3),
            nn.Conv1d(64, 32, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.AvgPool1d(kernel_size=avg_pool_kernel_size, stride=2),
            nn.BatchNorm1d(32),
            nn.ELU()
        )

        self.end_part = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(32, 1, kernel_size=1, stride=1, padding=padn),
            nn.AdaptiveAvgPool1d(output_size=1)
        )

    def forward(self, x):
        x = x - torch.mean(x, dim=-1).unsqueeze(dim=-1)
        x = self.first_part(x)
        x = self.middle_part(x)
        x = self.end_part(x)

        # output shape [1, 1, 1]
        return x


# --------------------------------------------------------------------------------------------------------------------
# CNN + LSTM - RateProbEstNet
# --------------------------------------------------------------------------------------------------------------------
class InceptionBlock(nn.Module):
    """
    Performs 5 parallel convolution with different kernel sizes and feed results in 5 channels
    """
    def __init__(self):
        super().__init__()
        kernels = [5, 21, 41, 61, 81]
        pads = [(x-1)//2 for x in kernels]

        self.conv_list = nn.ModuleList(
            [nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=p) for k, p in zip(kernels, pads)]
        )

    def forward(self, x):
        out = []
        for i in range(len(self.conv_list)):
            out.append(self.conv_list[i](x))

        out = tr.cat(out, dim=1)
        return out


class CNNBlock(nn.Module):
    def __init__(self):
        super().__init__()

        max_pool_kernel_size = 5
        conv_kernel_size = 21
        padn = (conv_kernel_size-1)//2

        self.first_part = nn.Sequential(
            nn.BatchNorm1d(5),
            nn.Dropout(0.1),

            nn.Conv1d(5, 32, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(32),
            nn.ELU(),

            nn.Dropout(0.1),
            nn.Conv1d(32, 32, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(32),
            nn.ELU(),

            nn.Dropout(0.1),
            nn.Conv1d(32, 64, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU()
        )

        self.middle_part = nn.Sequential(
            nn.Dropout(0.15),
            nn.Conv1d(64, 128, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.Dropout(0.15),
            nn.Conv1d(128, 128, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(64),
            nn.ELU(),

            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.BatchNorm1d(64),
            nn.ELU(),

            nn.Dropout(0.3),
            nn.Conv1d(64, 32, kernel_size=conv_kernel_size, stride=1, padding=padn),
            nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU()
        )

        self.end_part = nn.Sequential(
            nn.Dropout(0.35),
            nn.Conv1d(32, 5, kernel_size=1, stride=1, padding=0),
            nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(5),
            nn.ELU()
        )

    def forward(self, x):
        x = self.first_part(x)
        x = self.middle_part(x)
        x = self.end_part(x)
        # output shape [1, 5, 64]

        N, C, L = x.shape
        x = x.view(N, 1, C*L)
        # [1, 1, 320]
        return x


class RateProbLSTMCNN(nn.Module):
    def __init__(self, n_out=2):
        """
        :param n_out: number of estimated parameters, in cas of prob. output layer 2, or single output layer n_out=1
        """
        super().__init__()

        self.n_layers = 2
        self.n_hid = 80
        self.n_out = n_out

        self.inception_block = InceptionBlock()
        self.cnn_block = CNNBlock()

        self.lstm_layer1 = nn.LSTM(input_size=128, hidden_size=self.n_hid, num_layers=self.n_layers, dropout=0.2)
        self.lstm_layer2 = nn.LSTM(input_size=400, hidden_size=self.n_hid, num_layers=self.n_layers, dropout=0.5)

        self.linear = nn.Linear(80, self.n_out)

    def init_hidden(self, bsz):
        """
        Returns initial hidden state and hidden cell values
        """
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, bsz, self.n_hid),
                weight.new_zeros(self.n_layers, bsz, self.n_hid))

    def forward(self, x, h1=None, h2=None):
        # convolution stream
        x1 = self.inception_block(x)
        x1 = self.cnn_block(x1)

        # lstm stream
        self.lstm_layer1.flatten_parameters()
        if h1 is None:
            x2, h1 = self.lstm_layer1(x)
        else:
            x2, h1 = self.lstm_layer1(x, h1)

        x = tr.cat((x1, x2), dim=2)
        # torch.Size([8, 1, 400])

        # last part
        self.lstm_layer2.flatten_parameters()
        if h2 is None:
            x, h2 = self.lstm_layer2(x)
        else:
            x, h2 = self.lstm_layer2(x, h2)

        x = self.linear(x.view(-1, 80))

        if self.n_out == 2:
            x[:, 1] = F.elu(x[:, 1]) + 1.  # sigmas must have positive!

        for h in h1:
            h.detach_()
        for h in h2:
            h.detach_()

        return x, h1, h2


if __name__ == '__main__':
    def deepphys_test():
        model = DeepPhys()
        A = tr.randn(1, 3, 36, 36)
        out = model(A, A)
        print(out.shape, out)

    # deepphys_test()               # OK!

    def physnet_test():
        model = PhysNetED().to('cuda')
        model.eval()
        with tr.no_grad():
            x = tr.rand(1, 3, 128, 128, 128).to('cuda')
            out = model(x)
        print(out.shape)

    # physnet_test()                  # OK!

    def inception_test():
        incept_block = InceptionBlock()
        x = tr.randn(10, 1, 128)
        out = incept_block(x)
        cnnblock = CNNBlock()
        out = cnnblock(out)
        print(out.shape)

    # inception_test()        # OK

    def cnnblock_test():
        cnnblock = CNNBlock()
        x = tr.randn(10, 5, 128)
        out = cnnblock(x)
        print(out.shape)

    # cnnblock_test()

    def rateproblstmcnn_test():
        model = RateProbLSTMCNN(1)
        x = tr.randn(80, 1, 128)
        out, _, _ = model(x)
        print(out.shape)


    rateproblstmcnn_test()
