from src.archs import *
from src.dset import *
from src.errfuncs import *

import os
import time
import h5py
import numpy as np
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
tr = torch


def eval_model(model, testloader, criterion, oname):
    total_loss = []
    result = []
    for inputs, targets in tqdm(testloader):

        # Copy data to device
        for count, item in enumerate(inputs):
            inputs[count] = item.to(device)
        targets = targets.to(device)

        with tr.no_grad():
            outputs = model(*inputs).squeeze()
            # print(f'outputs.shape: {outputs.shape}')
            loss = criterion(outputs, targets)
            print(f'Current loss: {loss.item()}')

        # save network output
        result.extend(outputs.data.cpu().numpy()[:].tolist())
        print(f'List length: {len(result)}')

        total_loss.append(loss.item())

    total_loss = np.nanmean(total_loss)
    print(f'\n------------------------\nTotal loss: {total_loss}\n-----------------------------')
    np.savetxt(f'outputs/{oname}', np.array(result))
    print('Result saved!')


if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='DeepPhys, PhysNet')
    parser.add_argument('loss', type=str, help='Loss function: L1, RMSE, MSE, NegPea, SNR, Gauss, Laplace')
    parser.add_argument('data', type=str, help='path to .hdf5 file containing data')
    parser.add_argument("weights", type=str, help="if specified starts from checkpoint model")

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument("--ofile_name", type=str, help="output file name")
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during generation')
    parser.add_argument('--crop', type=bool, default=True, help='crop baby with yolo (preprocessing step)')
    parser.add_argument('--phase_shift', type=int, default=0, help='phase shift for reference signal')

    args = parser.parse_args()

    # --------------------------------------
    # Dataset and dataloader construction
    # --------------------------------------
    loader_device = None    # if multiple workers yolo works only on cpu
    if args.n_cpu == 0:
        loader_device = torch.device('cuda')
    else:
        loader_device = torch.device('cpu')

    testset = trainset = None
    if args.model == 'PhysNet':
        # chose label type for specific loss function
        if args.loss == 'SNR':
            ref_type = 'PulseNumerical'
        else:
            ref_type = 'PPGSignal'

        testset = Dataset4DFromHDF5(args.data,
                                    labels=(ref_type,),
                                    device=loader_device,
                                    crop=args.crop,
                                    augment=False,
                                    augment_freq=False)

    elif args.model == 'DeepPhys':
        testset = DatasetDeepPhysHDF5(args.data,
                                      device=loader_device,
                                      shift=args.phase_shift,
                                      crop=args.crop,
                                      augment=False)
    else:
        print('Error! No such model.')
        exit(666)

    testloader = DataLoader(testset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.n_cpu,
                            pin_memory=True)

    # --------------------------
    # Load model
    # --------------------------
    model = None
    if args.model == 'DeepPhys':
        model = DeepPhys()
    elif args.model == 'PhysNet':
        model = PhysNetED()
    else:
        print('\nError! No such model. Choose from: DeepPhys, PhysNet')
        exit(666)

    # Use multiple GPU if there are!
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = tr.nn.DataParallel(model)

    # If there are pretrained weights, initialize model
    model.load_state_dict(tr.load(args.weights))

    # Copy model to working device
    model = model.to(device)
    model.eval()

    # --------------------------
    # Define loss function
    # ---------------------------
    # 'L1, MSE, NegPea, SNR, Gauss, Laplace'
    loss_fn = None
    if args.loss == 'L1':
        loss_fn = nn.L1Loss()
    elif args.loss == 'MSE':
        loss_fn = nn.MSELoss()
    elif args.loss == 'NegPea':
        loss_fn = NegPeaLoss()
    elif args.loss == 'SNR':
        loss_fn = SNRLoss()
    elif args.loss == 'Gauss':
        loss_fn = GaussLoss()
    elif args.loss == 'Laplace':
        loss_fn = LaplaceLoss()
    else:
        print('\nError! No such loss function. Choose from: L1, MSE, NegPea, SNR, Gauss, Laplace')
        exit(666)

    # -------------------------------
    # Evaluate model
    # -------------------------------
    eval_model(model, testloader, criterion=loss_fn, oname=args.ofile_name)

    print('Succefully finished!')
