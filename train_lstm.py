from comet_ml import Experiment

from src.archs import PhysNetED, RateProbLSTMCNN
from src.errfuncs import LaplaceLoss, GaussLoss
from src.dset import Dataset4DFromHDF5

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
tr = torch


def train_model(models, dataloaders, criterion, optimizers, opath, num_epochs=35):
    val_loss_history = []
    train_loss_history = []

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        experiment.set_epoch(epoch)

        # Each epoch has a training and validation phase
        phases = ['train', 'val']
        for phase in phases:
            running_loss = 0.0
            if phase == 'train':
                for i in range(len(models)):
                    models[i].train()  # Set model to training mode -> activate droput layers and batch norm
            else:
                for i in range(len(models)):
                    models[i].eval()  # Set model to evaluate mode

            # Iterate over data.
            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # zero the parameter gradients
                for optimizer in optimizers:
                    optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Signal extraction
                    signals = models[0](inputs).view(-1, 1, 128)
                    # Rate estimation
                    rates, _, _ = models[1](signals)
                    loss = criterion(rates.view(-1, 1, 2), targets.view(-1, 1))
                    if phase == 'train':
                        loss.backward()
                        optimizers[0].step()
                        optimizers[1].step()

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
        for i, model in enumerate(models):
            torch.save(model.state_dict(), f'checkpoints/{opath}/model{i}_ep{epoch}.pt')
        print()


if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, nargs='+', help='DeepPhys, PhysNet, RateProbEst')
    parser.add_argument('--loss', type=str, help='L1, MSE, NegPea, SNR, Gauss, Laplace')
    parser.add_argument('--lr', type=float, nargs='+', default=1e-4, help='learning rate')
    parser.add_argument('--data', type=str, help='path to .hdf5 file containing data')
    parser.add_argument('--intervals', type=int, nargs='+', help='indices: train_start, train_end, val_start, val_end, shift_idx')
    parser.add_argument('--logger_name', type=str, help='project name for commet ml experiment')

    parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--checkpoint_dir", type=str, help="checkpoints will be saved in this directory")
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during generation')
    parser.add_argument('--crop', type=bool, default=False, help='crop baby with yolo (preprocessing step)')

    args = parser.parse_args()

    # Add the following code anywhere in your machine learning file
    experiment = Experiment(api_key="hs2nruoKow2CnUKisoeHccvh7", project_name=args.logger_name, workspace="terbed")

    hyper_params = {
        "model": args.model,
        "pretrained_weights": args.pretrained_weights,
        "checkpoint_dir": args.checkpoint_dir,
        "loss_fn": args.loss,
        "batch_size": args.batch_size,
        "n_workers": args.n_cpu,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "database": args.data,
        "intervals": args.intervals,
        "crop": args.crop,
        "img_augm": True,
        "freq_augm": True
    }

    experiment.log_parameters(hyper_params)

    # Fix random seed for reproducability
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ---------------------------------------
    # Construct datasets
    # ---------------------------------------
    ref_type = 'PulseNumerical'
    trainset = Dataset4DFromHDF5(args.data,
                                 labels=(ref_type,),
                                 device=torch.device('cpu'),
                                 start=args.intervals[0], end=args.intervals[1],
                                 crop=args.crop,
                                 augment=False,
                                 augment_freq=False,
                                 D=180,
                                 ccc=False
                                 )

    testset = Dataset4DFromHDF5(args.data,
                                labels=(ref_type,),
                                device=torch.device('cpu'),
                                start=args.intervals[2], end=args.intervals[3],
                                crop=args.crop,
                                augment=False,
                                augment_freq=False)

    # -------------------------
    # Construct DataLoaders
    # -------------------------
    trainloader = DataLoader(trainset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.n_cpu,
                             pin_memory=True,
                             collate_fn=trainset.collate_fn)

    testloader = DataLoader(testset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.n_cpu,
                            pin_memory=True)

    dataloaders_ = {'train': trainloader, 'val': testloader}
    print('\nDataLoaders successfully constructed!')

    # --------------------------
    # Load model
    # --------------------------
    models_ = [PhysNetED(), RateProbLSTMCNN()]

    # ----------------------------------
    # Set up training
    # ---------------------------------
    # Use multiple GPU if there are!
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        for i in range(len(models_)):
            models_[i] = tr.nn.DataParallel(models_[i])

    # If there are pretrained weights, initialize model
    if args.pretrained_weights:
        models_[0].load_state_dict(tr.load(args.pretrained_weights))
        print('\nPre-trained weights are loaded for PhysNet!')

    # Copy model to working device
    for i in range(len(models_)):
        models_[i] = models_[i].to(device)

    # --------------------------
    # Define loss function
    # ---------------------------
    # 'L1, MSE, NegPea, SNR, Gauss, Laplace'
    loss_fn = None
    if args.loss == 'Gauss':
        loss_fn = GaussLoss()
    elif args.loss == 'Laplace':
        loss_fn = LaplaceLoss()
    else:
        print('\nError! No such loss function. Choose from: Gauss, Laplace')
        exit(666)

    # ----------------------------
    # Initialize optimizer
    # ----------------------------
    opts = []
    for i, model in enumerate(models_):
        opts.append(optim.AdamW(model.parameters(), lr=args.lr[i]))

    # -----------------------------
    # Start training
    # -----------------------------
    train_model(models_, dataloaders_, criterion=loss_fn, optimizers=opts, opath=args.checkpoint_dir, num_epochs=args.epochs)

    experiment.end()

    print('\nTraining is finished without flaw!')
