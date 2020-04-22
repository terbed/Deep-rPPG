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


def eval_model(models, testloader, criterion, oname):
    total_loss = []
    result = []
    signal = []
    ref = []
    for inputs, targets in tqdm(testloader):

        # Copy data to device
        for count, item in enumerate(inputs):
            inputs[count] = item.to(device)
        targets = targets.to(device)

        with tr.no_grad():
            if len(models) == 1:
                outputs = models[0](inputs).squeeze()
                # print(f'outputs.shape: {outputs.shape}')

                if criterion is not None:
                    loss = criterion(outputs, targets)
                    print(f'Current loss: {loss.item()}')

                # save network output
                result.extend(outputs.data.cpu().numpy().flatten().tolist())
                # print(f'List length: {len(result)}')
            elif len(models) == 2:
                signals = models[0](inputs).view(-1, 1, 128)
                n_batch = signals.shape[0]
                if n_batch > 1:
                    rates = models[1](signals).view(-1, 2)
                    targets = targets.squeeze()
                    print(f'in inference targets.shape: {targets.shape}')
                    # print(targets)
                    result.extend(rates.data.cpu().numpy().tolist())
                    signal.extend(signals.data.cpu().numpy().flatten().tolist())
                    ref.extend(targets.data.cpu().numpy().tolist())
                else:   # Use dropout during eval to have model uncertainty :TODO bring into life this function
                    models[1].eval()
                    for m in models[1].modules():
                        if m.__class__.__name__.startswith('Dropout'):
                            m.train()
                    signals = signals.repeat(10, 1, 128)
                    rates = models[1](signals).view(-1, 2)
                    std = tr.sum(rates.std(dim=0))
                    m = rates.mean(dim=0)
                    res = [m[0].item(), (m[1]+std).item()]

                    result.extend(res)
                    signal.extend(signals.data.cpu().numpy().flatten().tolist())
                    ref.extend(targets.data.cpu().numpy().tolist())

        if criterion is not None:
            total_loss.append(loss.item())

    if criterion is not None:
        total_loss = np.nanmean(total_loss)
        print(f'\n------------------------\nTotal loss: {total_loss}\n-----------------------------')

    if len(models) == 1:
        np.savetxt(f'outputs/{oname}', np.array(result))
    elif len(models) == 2:
        result = np.array(result)
        ref = np.array(ref)
        signal = np.array(signal)
        with h5py.File(f'outputs/{oname}', 'w') as db:
            db.create_dataset('reference', shape=ref.shape, dtype=np.float32, data=ref)
            db.create_dataset('signal', shape=signal.shape, dtype=np.float32, data=signal)
            db.create_dataset('rates', shape=result.shape, dtype=np.float32, data=result)
    print('Result saved!')


if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, nargs='+', help='DeepPhys, PhysNet, RateProbEst, RateEst')
    parser.add_argument('--data', type=str, help='path to benchmark .hdf5 file containing data')
    parser.add_argument('--interval', type=int, nargs='+',
                        help='indices: val_start, val_end, shift_idx; if not given -> whole dataset')
    parser.add_argument("--weights", type=str, nargs='+', help="model weight paths")

    parser.add_argument('--loss', type=str, default=None, help='Loss function: L1, RMSE, MSE, NegPea, SNR, Gauss, Laplace')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument("--ofile_name", type=str, help="output file name")
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during generation')
    parser.add_argument('--crop', type=bool, default=False, help='crop baby with yolo (preprocessing step)')
    parser.add_argument('--phase_shift', type=int, default=0, help='phase shift for reference signal')

    args = parser.parse_args()
    start_idx = end_idx = None
    if args.interval:
        start_idx, end_idx = args.interval

    # --------------------------------------
    # Dataset and dataloader construction
    # --------------------------------------
    loader_device = None    # if multiple workers yolo works only on cpu
    if args.n_cpu == 0:
        loader_device = torch.device('cuda')
    else:
        loader_device = torch.device('cpu')

    testset = trainset = None
    if args.model[0] == 'PhysNet':
        print('Constructing data loader for PhysNet architecture...')
        # chose label type for specific loss function
        if args.loss == 'SNR' or args.loss == 'Laplace' or args.loss == 'Gauss':
            ref_type = 'PulseNumerical'
            print('\nPulseNumerical reference type chosen!')
        elif args.loss == 'L1' or args.loss == 'MSE' or args.loss == 'NegPea':
            ref_type = 'PPGSignal'
            print('\nPPGSignal reference type chosen!')
        else:
            ref_type = 'PulseNumerical'
            print('\nPulseNumerical reference type chosen!')

        testset = Dataset4DFromHDF5(args.data,
                                    labels=(ref_type,),
                                    device=loader_device,
                                    start=start_idx, end=end_idx,
                                    crop=args.crop,
                                    augment=False,
                                    augment_freq=False)

    elif args.model[0] == 'DeepPhys':
        if args.interval:
            phase_shift = args.interval[4] if len(args.interval) == 5 else 0            # init phase shift parameter
        else:
            phase_shift = 0

        testset = DatasetDeepPhysHDF5(args.data,
                                      device=loader_device,
                                      start=start_idx, end=end_idx,
                                      shift=phase_shift,
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
    models = []
    if len(args.model) == 1:
        if args.model[0] == 'DeepPhys':
            models.append(DeepPhys())
        elif args.model[0] == 'PhysNet':
            models.append(PhysNetED())
        else:
            print('\nError! No such model. Choose from: DeepPhys, PhysNet')
            exit(666)
    elif len(args.model) == 2:
        # signal extractor model
        models.append(PhysNetED())
        # rate estimator model
        if args.model[1] == 'RateProbEst':
            models.append(RateProbEst())
        elif args.model[1] == 'RateEst':
            models.append(RateEst())
        else:
            print('\nNo such estimator model! Choose from: RateProbEst, RateEst')
            exit(666)

    # Use multiple GPU if there are!
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        for i in range(len(models)):
            models[i] = tr.nn.DataParallel(models[i])

    # If there are pretrained weights, initialize model
    for i, model in enumerate(models):
        model.load_state_dict(tr.load(args.weights[i]))

    # Copy model to working device
    for i in range(len(models)):
        models[i] = models[i].to(device)
        models[i].eval()

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
        print('\nHey! No such loss function. Choose from: L1, MSE, NegPea, SNR, Gauss, Laplace')
        print('Inference with no loss function')

    # -------------------------------
    # Evaluate model
    # -------------------------------
    eval_model(models, testloader, criterion=loss_fn, oname=args.ofile_name)

    print('Successfully finished!')
