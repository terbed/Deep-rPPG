"""
PyTorch Dataset classes for dataloader
"""
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import h5py
import random
from torchvision.transforms import RandomRotation, ToPILImage, ToTensor, ColorJitter
import torchvision.transforms.functional as TF

from yolo.detect import babybox
from yolo.models import *
from yolo.utils import *
tr = torch


class Dataset4DFromHDF5(Dataset):
    """
        Dataset class for PhysNet neural network.
    """

    def __init__(self, path: str, labels: tuple, device, start=None, end=None, D=128, C=3, H=128, W=128, crop=True,
                 augment=False,  augment_freq=False):
        """

        :param path: Path to hdf5 file
        :param labels: tuple of label names to use (e.g.: ('pulseNumerical', 'resp_signal') or ('pulse_signal', ) )
            Note that the first label must be the pulse rate if it is present!
        """
        self.device = device
        self.D = D
        self.H = H
        self.W = W
        self.C = C
        self.crop = crop
        self.augment = augment
        self.augment_frq = augment_freq

        # ---------------------------
        # Augmentation variables
        # ---------------------------
        self.flip_p = None
        self.rot = None
        self.color_transform = None
        self.freq_scale_fact = None

        # -----------------------------
        # Init baby cropper
        # -----------------------------
        if self.crop:
            model_def = 'yolo/config/yolov3-custom.cfg'
            weight_path = 'yolo/weights/yolov3_ckpt_42.pth'

            self.yolo = Darknet(model_def).to(self.device)
            self.yolo.load_state_dict(torch.load(weight_path))
            self.yolo.eval()
            print("YOLO network is initialized and ready to work!")

        # -----------------------------
        # Open database
        # -----------------------------
        self.db_path = path
        db = h5py.File(path, 'r')
        frames = db['frames']
        db_labels = db['references']

        # Append all required label from database
        self.labels = []
        self.label_names = labels
        for label in labels:
            self.labels.append(db_labels[label][:])

        (self.n, H, W, C) = frames.shape
        print(f'\nNumber of frames in the whole dataset: {self.n}')

        if start is not None:
            self.n = end - start
            self.begin = start
        else:
            self.begin = 0

        print(f'\nNumber of images in the chosen interval: {self.n}')
        print(f'Size of an image: {H} x {W} x {C}')

        self.num_samples = ((self.n - 64) // self.D) - 1
        print(f'Number of samples in the dataset: {self.num_samples}\n')
        db.close()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        d = self.D

        if self.augment:
            # Set up the same image transforms for the chunk
            self.flip_p = random.random()
            self.hflip_p = random.random()
            self.rot = RandomRotation.get_params((0, 90))
            self.color_transform = ColorJitter.get_params(brightness=(0.7, 1.3),
                                                          contrast=(0.8, 1.2),
                                                          saturation=(0.8, 1.2),
                                                          hue=(0, 0))

        # -------------------------------
        # Set up frequency augmentation
        # -------------------------------
        if self.augment_frq:
            self.freq_scale_fact = np.around(np.random.uniform(0.7, 1.4), decimals=1)
            d = int(np.around(self.D * self.freq_scale_fact))

        # Construct target signals
        targets = []
        for label in self.labels:
            label_segment = label[self.begin + idx * self.D: self.begin + idx * self.D + self.D]
            targets.append(tr.from_numpy(label_segment).type(tr.FloatTensor))

        # Construct networks input
        video = tr.empty(self.C, d, self.H, self.W, dtype=tr.float)

        with h5py.File(self.db_path, 'r') as db:
            frames = db['frames']
            # ------------------------------------------------------
            # Calculate bounding box for the baby for this segment
            # ------------------------------------------------------
            if self.crop:
                first_frame = frames[self.begin + idx * self.D, :]
                x1, y1, x2, y2 = babybox(self.yolo, first_frame, self.device)

            # -------------------------------
            # Fill video with frames
            # -------------------------------
            # conv3d input: N x C x D x H X W
            for i in range(d):
                img = frames[self.begin + idx * self.D + i, :]
                # Crop baby from image
                if self.crop:
                    img = img[y1:y2, x1:x2, :]
                # Downsample cropped image
                img = cv2.resize(img, (self.H, self.W), interpolation=cv2.INTER_AREA)

                if self.augment:
                    # img = cv2.convertScaleAbs(img, alpha=(255.0 / np.max(img))) # convert to uint8
                    img = ToPILImage()(img)
                    if self.flip_p > 0.5:
                        img = TF.vflip(img)
                    if self.flip_p > 0.5:
                        img = TF.hflip(img)
                    # img = TF.rotate(img, self.rot)
                    img = self.color_transform(img)
                    img = ToTensor()(img)  # uint8 H x W x C -> torch image: float32 [0, 1] C X H X W
                else:
                    img = ToTensor()(img)  # uint8 H x W x C -> torch image: float32 [0, 1] C X H X W

                img = tr.sub(img, tr.mean(img, (1, 2)).view(3, 1, 1))  # Color channel centralization
                video[:, i, :] = img

        # ------------------------------
        # Apply frequency augmentation
        # ------------------------------
        if self.augment_frq:
            # edit video
            resampler = torch.nn.Upsample(size=(self.D, self.H, self.W), mode='trilinear')
            video = resampler(video.unsqueeze(0)).squeeze()

            # edit labels
            for counter, name in enumerate(self.label_names):
                if name == 'PulseNumerical':
                    targets[counter] = targets[counter] * self.freq_scale_fact
                    # select the most frequent rate value for each batch and convert to Hz
                    targets[counter] = tr.mode(targets[counter])[0] / 60.
                elif name == 'PPGSignal':
                    segment = tr.from_numpy(
                        self.labels[counter][self.begin + idx * self.D: self.begin + idx * self.D + d])
                    resampler = torch.nn.Upsample(size=(self.D,), mode='linear')
                    segment = resampler(segment.view(1, 1, -1))
                    segment = segment.squeeze()
                    targets[counter] = segment

            sample = ((video,), *targets)
        else:
            sample = ((video,), *targets)

        # Video shape: C x D x H X W
        return sample


class DatasetDeepPhysHDF5(Dataset):
    """
        Dataset class for training network.
    """

    def __init__(self, path: str, device, start=None, end=None, shift=0, crop=True, augment=False):
        """

        :param path: Path to hdf5 file
        :param labels: tuple of label names to use (e.g.: ('pulseNumerical', 'resp_signal') or ('pulse_signal', ) )
            Note that the first label must be the pulse rate if it is present!
        """
        from src.utils import ReferenceProcessor

        self.H = 36
        self.W = 36
        self.C = 3
        self.augment = augment
        self.device = device
        self.crop = crop

        # -----------------------------
        # Augmentation variables
        # ----------------------------
        self.flip_p = None
        self.rot = None
        self.color_transform = None

        # -----------------------------
        # Init baby cropper
        # -----------------------------
        if crop:
            model_def = 'yolo/config/yolov3-custom.cfg'
            weight_path = 'yolo/weights/yolov3_ckpt_42.pth'

            self.yolo = Darknet(model_def).to(device)
            self.yolo.load_state_dict(torch.load(weight_path))
            self.yolo.eval()
            print("YOLO network is initialized and ready to work!")

        self.db_path = path
        db = h5py.File(path, 'r')
        frames = db['frames']
        db_labels = db['references']

        # Create derivated ppg label
        ppg_label = db_labels['PPGSignal']
        refproc = ReferenceProcessor(ppg_label[shift:])
        refproc.calculate()
        self.label = refproc.training_label

        (self.n, H, W, C) = frames.shape

        if start is not None:
            self.n = end - start
            self.begin = start
        else:
            self.begin = 0

        print(f'\nNumber of images in the dataset: {self.n}')
        print(f'Size of an image: {H} x {W} x {C}')

        self.num_samples = self.n - 1 - shift
        db.close()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.augment:
            # Set up the same image transforms for the chunk
            self.flip_p = random.random()
            self.hflip_p = random.random()
            self.rot = RandomRotation.get_params((0, 90))
            self.color_transform = ColorJitter.get_params(brightness=(0.7, 1.3),
                                                          contrast=(0.8, 1.2),
                                                          saturation=(0.8, 1.2),
                                                          hue=(0, 0))

        # Construct target signals
        target = tr.tensor(self.label[idx])

        # Construct networks input
        A = tr.empty(self.C, self.H, self.W, dtype=tr.float)
        M = tr.empty(self.C, self.H, self.W, dtype=tr.float)

        with h5py.File(self.db_path, 'r') as db:
            frames = db['frames']
            img1 = frames[idx, :, :, :]
            img2 = frames[idx + 1, :, :, :]

        # ----------------------------
        # Crop baby with yolo
        # ----------------------------
        if self.crop:
            x1, y1, x2, y2 = babybox(self.yolo, img1, self.device)
            img1 = img1[y1:y2, x1:x2, :]
            img2 = img2[y1:y2, x1:x2, :]

        # Downsample image
        img1 = cv2.resize(img1, (self.H, self.W), interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, (self.H, self.W), interpolation=cv2.INTER_CUBIC)

        if self.augment:
            img1 = ToPILImage()(img1)
            img2 = ToPILImage()(img2)
            if self.flip_p > 0.5:
                img1 = TF.vflip(img1)
                img2 = TF.vflip(img2)
            if self.flip_p > 0.5:
                img1 = TF.hflip(img1)
                img2 = TF.hflip(img2)

            img1 = TF.rotate(img1, self.rot)
            img2 = TF.rotate(img2, self.rot)

            img1 = self.color_transform(img1)
            img2 = self.color_transform(img2)

            img1 = tr.from_numpy(np.array(img1).astype(np.float32))
            img2 = tr.from_numpy(np.array(img2).astype(np.float32))
            img1 = img1.permute(2, 0, 1)
            img2 = img2.permute(2, 0, 1)
        else:
            img1 = tr.from_numpy(img1.astype(np.float32))
            img2 = tr.from_numpy(img2.astype(np.float32))
            # Swap axes because  numpy image: H x W x C | torch image: C X H X W
            img1 = img1.permute(2, 0, 1)
            img2 = img2.permute(2, 0, 1)

        # 2.) construct the normalized frame difference for motion stream input
        M = tr.div(img2 - img1, img1 + img2 + 1)  # +1 for numerical stability

        A = img1/255.  # convert image to [0, 1]
        A = tr.sub(A, tr.mean(A, (1, 2)).view(3, 1, 1))  # spatial intensity norm for each channel

        sample = ((A, M), target)

        # Video shape: C x D x H X W
        return sample
