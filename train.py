from comet_ml import Experiment

from src.archs import *
from src.errfuncs import *
from src.dset import *

import os
import sys
import time
import argparse

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
tr = torch


def train_model(model, dataloaders, criterion, optimizer, opath, num_epochs=35):
    val_loss_history = []
    train_loss_history = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        experiment.set_epoch(epoch)

        # Each epoch has a training and validation phase
        phases = ['train', 'val']
        for phase in phases:
            running_loss = 0.0
            if phase == 'train':
                model.train()  # Set model to training mode -> activate droput layers and batch norm
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for video, targets in dataloaders[phase]:

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(video).squeeze()
                    loss = criterion(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloaders[phase])
            print('{} Loss: {:.4f} '.format(phase, epoch_loss))

            if phase == 'val':
                val_loss_history.append(epoch_loss)
                with experiment.test():
                    experiment.log_metric("loss", epoch_loss, step=epoch)
            else:
                train_loss_history.append(epoch_loss)
                with experiment.train():
                    experiment.log_metric("loss", epoch_loss, step=epoch)

        experiment.log_epoch_end(epoch)
        torch.save(model.state_dict(), f'checkpoints/{opath}/ep_{epoch}.pt')
        print()

    return model, val_loss_history, train_loss_history


if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='DeepPhys, PhysNet')
    parser.add_argument('loss', type=str, help='L1, MSE, NegPea, SNR, Gauss, Laplace')
    parser.add_argument('data', type=str, help='path to .hdf5 file containing data')
    parser.add_argument('intervals', type=int, nargs='+', help='indices: train_start, train_end, val_start, val_end, shift_idx')
    parser.add_argument('logger_name', type=str, help='project name for commet ml experiment')

    parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--checkpoint_dir", type=str, help="checkpoints will be saved in this directory")
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during generation')
    parser.add_argument('--img_size', type=int, default=128, help='size of image')
    parser.add_argument('--time_depth', type=int, default=128, help='time depth for PhysNet')
    parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
    parser.add_argument('--crop', type=bool, default=True, help='crop baby with yolo (preprocessing step)')
    parser.add_argument('--img_augm', type=bool, default=True, help='image augmentation (flip, color jitter)')
    parser.add_argument('--freq_augm', type=bool, default=False, help='apply frequency augmentation')

    opt = parser.parse_args()

    # Add the following code anywhere in your machine learning file
    experiment = Experiment(api_key="hs2nruoKow2CnUKisoeHccvh7", project_name=opt.logger_name, workspace="terbed")

    hyper_params = {
        "model": opt.model,
        "pretrained_weights": opt.pretrained_weights,
        "checkpoint_dir": opt.checkpoint_dir,
        "loss_fn": opt.loss,
        "time_depth": opt.time_depth,
        "img_size": opt.img_size,
        "batch_size": opt.batch_size,
        "num_epochs": opt.epochs,
        "learning_rate": opt.lr,
        "database": opt.data,
        "intervals": opt.intervals,
        "crop": opt.crop,
        "img_augm": opt.img_augm,
        "freq_augm": opt.freq_augm
    }

    experiment.log_parameters(hyper_params)

    # Fix random seed for reproducability
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # --------------------------------------
    # Dataset and dataloader construction
    # --------------------------------------
    testset = trainset = None
    if opt.model == 'PhysNet':
        # chose label type for specific loss function
        if opt.loss == 'SNR':
            ref_type = 'PulseNumerical'
        else:
            ref_type = 'PPGSignal'

        trainset = Dataset4DFromHDF5(path=opt.data,
                                     labels=(ref_type,),
                                     device=device,
                                     start=opt.intervals[0], end=opt.intervals[1],
                                     crop=opt.crop,
                                     augment=opt.img_augm,
                                     augment_freq=opt.freq_augm)

        testset = Dataset4DFromHDF5(path=opt.data,
                                    labels=(ref_type,),
                                    device=device,
                                    start=opt.intervals[2], end=opt.intervals[3],
                                    crop=opt.crop,
                                    augment=False,
                                    augment_freq=False)

    elif opt.model == 'DeepPhys':
        phase_shift = opt.intervals[4] if len(opt.intervals) == 5 else 0            # init phase shift parameter
        trainset = DatasetDeepPhysHDF5(path=opt.data,
                                       device=device,
                                       start=opt.intervals[0], end=opt.intervals[1],
                                       shift=phase_shift,
                                       crop=opt.crop,
                                       augment=opt.img_augm)

        testset = DatasetDeepPhysHDF5(path=opt.data,
                                      device=device,
                                      start=opt.intervals[2], end=opt.intervals[3],
                                      shift=phase_shift,
                                      crop=opt.crop,
                                      augment=opt.img_augm)
    else:
        print('Error! No such model.')
        exit(666)

    # Construct DataLoaders
    trainloader = DataLoader(trainset,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             num_workers=opt.n_cpu,
                             pin_memory=True)

    testloader = DataLoader(testset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.n_cpu,
                            pin_memory=True
                            )

    dataloaders = {'train': trainloader, 'val': testloader}
    print('\nDataLoaders succesfully constructed!')

    # --------------------------
    # Load model
    # --------------------------
    model = None
    if opt.model == 'DeepPhys':
        model = DeepPhys()
    elif opt.model == 'PhysNet':
        model = PhysNetED()
    else:
        print('\nError! No such model. Choose from: DeepPhys, PhysNet')
        exit(666)

    # Use multiple GPU if there are!
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = tr.nn.DataParallel(model)

    # If there are pretrained weights, initialize model
    if opt.pretrained_weights:
        model.load_state_dict(tr.load(opt.pretrained_weights))

    # Copy model to working device
    model = model.to(device)

    # --------------------------
    # TODO: Define loss function
    # ---------------------------

