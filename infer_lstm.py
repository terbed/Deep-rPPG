from src.archs import PhysNetED, RateProbLSTMCNN
from src.dset import Dataset4DFromHDF5

import h5py
import numpy as np
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
tr = torch


def eval_model(models, testloader, oname, device):
    total_loss = []
    result = []
    signal = []
    ref = []
    h1 = h2 = None

    for inputs, targets in tqdm(testloader):
        with tr.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Signal extractor
            signals = models[0](inputs).view(-1, 1, 128)
            # Rate estimator
            rates, h1, h2 = models[1](signals, h1, h2)

            targets = targets.squeeze()

        # print(f'in inference targets.shape: {targets.shape}')
        # print(targets)
        result.extend(rates.data.cpu().numpy().tolist())
        signal.extend(signals.data.cpu().numpy().flatten().tolist())
        ref.extend(targets.data.cpu().numpy().reshape(-1, 1).tolist())

    result = np.array(result)
    ref = np.array(ref)
    signal = np.array(signal)
    with h5py.File(f'outputs/{oname}.h5', 'w') as db:
        db.create_dataset('reference', shape=ref.shape, dtype=np.float32, data=ref)
        db.create_dataset('signal', shape=signal.shape, dtype=np.float32, data=signal)
        db.create_dataset('rates', shape=result.shape, dtype=np.float32, data=result)

    print('Result saved!')


if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device_ = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device_)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path to benchmark .hdf5 file containing data')
    parser.add_argument('--n_out', type=int, default=2, help='Number of output parameters of tha rate network')
    parser.add_argument('--interval', type=int, nargs='+',
                        help='indices: val_start, val_end, shift_idx; if not given -> whole dataset')
    parser.add_argument("--weights", type=str, nargs='+', help="model weight paths")

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument("--ofile_name", type=str, help="output file name")
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during generation')
    parser.add_argument('--crop', type=bool, default=False, help='crop baby with yolo (preprocessing step)')

    args = parser.parse_args()
    start_idx = end_idx = None
    if args.interval:
        start_idx, end_idx = args.interval

    # ---------------------------------------
    # Construct datasets
    # ---------------------------------------
    ref_type = 'PulseNumerical'
    testset = Dataset4DFromHDF5(args.data,
                                labels=(ref_type,),
                                device=torch.device('cpu'),
                                start=start_idx, end=end_idx,
                                crop=args.crop,
                                augment=False,
                                augment_freq=False
                                )

    testloader_ = DataLoader(testset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.n_cpu,
                             pin_memory=True)

    # --------------------------
    # Load model
    # --------------------------
    models_ = [PhysNetED(), RateProbLSTMCNN(args.n_out)]

    # ----------------------------------
    # Set up training
    # ---------------------------------
    for i in range(len(models_)):
        models_[i] = tr.nn.DataParallel(models_[i])
        models_[i].load_state_dict(tr.load(args.weights[i], map_location=device_))

    # Use multiple GPU if there are!
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        for i in range(len(models_)):
            models_[i] = models_[i].module

    # Copy model to working device
    for i in range(len(models_)):
        models_[i] = models_[i].to(device_)

    # -------------------------------
    # Evaluate model
    # -------------------------------
    eval_model(models_, testloader_, oname=args.ofile_name, device=device_)

    print('Successfully finished!')
