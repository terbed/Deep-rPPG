import torch
import argparse


if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, nargs='+', help='DeepPhys, PhysNet, RateProbEst')
    parser.add_argument('--loss', type=str, help='L1, MSE, NegPea, SNR, Gauss, Laplace')
    parser.add_argument('--data', type=str, help='path to .hdf5 file containing data')
    parser.add_argument('--intervals', type=int, nargs='+', help='indices: train_start, train_end, val_start, val_end, shift_idx')
    parser.add_argument('--logger_name', type=str, help='project name for commet ml experiment')

    parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--checkpoint_dir", type=str, help="checkpoints will be saved in this directory")
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during generation')
    parser.add_argument('--img_size', type=int, default=128, help='size of image')
    parser.add_argument('--time_depth', type=int, default=128, help='time depth for PhysNet')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--crop', type=bool, default=True, help='crop baby with yolo (preprocessing step)')
    parser.add_argument('--img_augm', type=bool, default=False, help='image augmentation (flip, color jitter)')
    parser.add_argument('--freq_augm', type=bool, default=False, help='apply frequency augmentation')
    
    args = parser.parse_args()
    print(len(args.model))
    print(args.model)