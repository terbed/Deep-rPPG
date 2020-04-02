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
from src.utils import img2uint8, pad_to_square
tr = torch


class Dataset4DFromHDF5(Dataset):
    """
        Dataset class for PhysNet neural network.
    """

    def __init__(self, path: str, labels: tuple, device, start=None, end=None, D=128, C=3, H=128, W=128, crop=True,
                 augment=False,  augment_freq=False, ccc=True):
        """

        :param path: Path to hdf5 file
        :param labels: tuple of label names to use (e.g.: ('pulseNumerical', 'resp_signal') or ('pulse_signal', ) )
            Note that the first label must be the pulse rate if it is present!
        :param D: In case of using collate_fn in dataloader, set this to 180 -> D=180
        :param ccc: color channel centralization
        """
        self.device = device
        self.D = D
        self.H = H
        self.W = W
        self.C = C
        self.crop = crop
        self.augment = augment
        self.augment_frq = augment_freq
        self.ccc = ccc

        # ---------------------------
        # Augmentation variables
        # ---------------------------
        self.flip_p = None
        self.rot = None
        self.color_transform = None
        self.freq_scale_fact = None

        # -----------------------------
        # Open database
        # -----------------------------
        self.db_path = path
        db = h5py.File(path, 'r')
        frames = db['frames']
        db_labels = db['references']

        # check if there is bbox
        keys = db.keys()
        print(keys)
        self.is_bbox = 'bbox' in keys
        if self.is_bbox:
            print('\nBounding boxes found in database, lets use them instead of YOLO!')

        # -----------------------------
        # Init baby cropper
        # -----------------------------
        if self.crop and not self.is_bbox:
            model_def = 'yolo/config/yolov3-custom.cfg'
            weight_path = 'yolo/weights/yolov3_ckpt_42.pth'

            self.yolo = Darknet(model_def).to(self.device)
            self.yolo.load_state_dict(torch.load(weight_path))
            self.yolo.eval()
            print("YOLO network is initialized and ready to work!")

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

        # ------------------------------------
        # Set up video augmentation parameters
        # ------------------------------------
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

        # ---------------------------
        # Construct target signals
        # ----------------------------
        targets = []
        for count, label in enumerate(self.labels):
            label_segment = label[self.begin + idx * self.D: self.begin + idx * self.D + self.D]
            label_segment = tr.from_numpy(label_segment).type(tr.FloatTensor)
            # If numerical select mode value
            if self.label_names[count] == 'PulseNumerical':
                # label_segment = tr.mode(label_segment.squeeze())[0] / 60.
                # label_segment = tr.mean(label_segment.squeeze()) / 60.
                label_segment = label_segment.squeeze() / 60.
            targets.append(label_segment)

        # ----------------------------
        # Construct networks input
        # -----------------------------
        video = tr.empty(self.C, d, self.H, self.W, dtype=tr.float)
        with h5py.File(self.db_path, 'r') as db:
            frames = db['frames']

            # Calculate or load bounding box for the baby for this segment
            if self.crop and self.is_bbox:
                bbox = db['bbox'][self.begin + idx * self.D, :]
                x1, x2, y1, y2 = self.bbox_checker(bbox[2], bbox[3], bbox[0], bbox[1])
            elif self.crop and not self.is_bbox:
                first_frame = frames[self.begin + idx * self.D, :]
                x1, y1, x2, y2 = babybox(self.yolo, first_frame, self.device)

            # -------------------------------
            # Fill video with frames
            # -------------------------------
            # conv3d input: N x C x D x H X W
            for i in range(d):
                img = frames[self.begin + idx * self.D + i, :]
                img = img2uint8(img)
                # Crop baby from image
                if self.crop:
                    img = img[y1:y2, x1:x2, :]
                    # pad to square
                    # img = ToTensor()(img)
                    # img, _ = pad_to_square(img, 0)
                    # img = np.array(ToPILImage()(img))
                # Downsample cropped image
                img = cv2.resize(img, (self.H, self.W), interpolation=cv2.INTER_AREA)
                # Augment if needed and transform to tensor
                if self.augment:
                    img = self.img_transforms(img)
                else:
                    img = ToTensor()(img)  # uint8 H x W x C -> torch image: float32 [0, 1] C X H X W
                    if self.ccc:
                        img = tr.sub(img, tr.mean(img, (1, 2)).view(3, 1, 1))  # Color channel centralization

                video[:, i, :] = img

        # ------------------------------
        # Apply frequency augmentation
        # ------------------------------
        if self.augment_frq:
            sample = self.freq_augm(d, idx, targets, video)
        else:
            if self.ccc:
                targets[0] = tr.mean(targets[0])
            sample = (video, *targets)

        # Video shape: C x D x H X W
        return sample

    def collate_fn(self, batch):
        """
        This function applies the same augmentation for each batch to result in an LSTM sequence
        """
        videos, targets = list(zip(*batch))

        # Set up the same image transforms for the given number of batches
        self.flip_p = random.random()
        self.hflip_p = random.random()
        self.color_transform = ColorJitter.get_params(brightness=(0.5, 1.3),
                                                      contrast=(0.8, 1.2),
                                                      saturation=(0.8, 1.2),
                                                      hue=(0, 0))

        # set up parameters for frequency augmentation
        desired_d = 128
        self.freq_scale_fact = np.around(np.random.uniform(0.7, 1.3), decimals=1)
        d = int(np.around(desired_d * self.freq_scale_fact))

        # -----------------------------
        # Augment labels accordingly
        # -----------------------------
        targets = tr.stack([tr.mean(target[:d]) * self.freq_scale_fact for target in targets]).unsqueeze(1)
        # print(f'Targets: {targets.shape}')

        # -------------------------------------
        # Augment video same way for each batch
        # -------------------------------------
        # Frequency augmentation
        # print(len(videos))
        videos = tr.stack(videos)
        # print(f'videos: {videos.shape}')
        resampler = torch.nn.Upsample(size=(desired_d, self.H, self.W), mode='trilinear', align_corners=False)
        videos = resampler(videos[:, :, 0:desired_d, :, :])
        # Image augmentation
        for b in range(videos.shape[0]):
            for d in range(videos.shape[2]):
                videos[b, :, d, :, :] = self.img_transforms(videos[b, :, d, :, :])

        return videos, targets

    def freq_augm(self, d, idx, targets, video):
        # edit video
        resampler = torch.nn.Upsample(size=(self.D, self.H, self.W), mode='trilinear', align_corners=False)
        video = resampler(video.unsqueeze(0)).squeeze()
        # edit labels
        for counter, name in enumerate(self.label_names):
            if name == 'PulseNumerical':
                targets[counter] = targets[counter] * self.freq_scale_fact
            elif name == 'PPGSignal':
                segment = tr.from_numpy(
                    self.labels[counter][self.begin + idx * self.D: self.begin + idx * self.D + d])
                resampler = torch.nn.Upsample(size=(self.D,), mode='linear', align_corners=False)
                segment = resampler(segment.view(1, 1, -1))
                segment = segment.squeeze()
                targets[counter] = segment
        sample = ((video,), *targets)
        return sample

    def img_transforms(self, img):
        img = ToPILImage()(img)
        if self.flip_p > 0.5:
            img = TF.vflip(img)
        if self.flip_p > 0.5:
            img = TF.hflip(img)
        # img = TF.rotate(img, self.rot)
        img = self.color_transform(img)
        img = ToTensor()(img)  # uint8 H x W x C -> torch image: float32 [0, 1] C X H X W

        img = tr.sub(img, tr.mean(img, (1, 2)).view(3, 1, 1))  # Color channel centralization
        return img

    def bbox_checker(self, x1, x2, y1, y2):
        # check to be inside image size
        if y2 > self.H:
            y2 = self.H
        if x2 > self.W:
            x2 = self.W
        if y1 < 0:
            y1 = 0
        if x1 < 0:
            x1 = 0
        # check validity
        if y2 - y1 < 1 or x2 - x1 < 1:
            y1 = x1 = 0
            y2, x2 = self.W, self.H
        return x1, x2, y1, y2


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

        self.db_path = path
        db = h5py.File(path, 'r')
        frames = db['frames']
        db_labels = db['references']

        # check if there is bbox
        keys = db.keys()
        print(keys)
        self.is_bbox = 'bbox' in keys
        if self.is_bbox:
            print('\nBounding boxes found in database, lets use them instead of YOLO!')

        # -----------------------------
        # Init baby cropper
        # -----------------------------
        if crop and not self.is_bbox:
            model_def = 'yolo/config/yolov3-custom.cfg'
            weight_path = 'yolo/weights/yolov3_ckpt_42.pth'

            self.yolo = Darknet(model_def).to(device)
            self.yolo.load_state_dict(torch.load(weight_path))
            self.yolo.eval()
            print("YOLO network is initialized and ready to work!")

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

        # self.num_samples = self.n - 1 - shift
        # To be in line with physnet
        tmp = ((self.n - 64) // 128)
        self.num_samples = tmp * 128 - 1 - shift
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
            # self.rot = RandomRotation.get_params((0, 90))
            self.color_transform = ColorJitter.get_params(brightness=(0.3, 1.5),
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
            if self.crop and self.is_bbox:
                bbox = db['bbox'][idx, :]
                y1, y2, x1, x2 = bbox[0], bbox[1], bbox[2], bbox[3]

                # check to be inside image size
                if y2 > img1.shape[0]:
                    y2 = img1.shape[0]
                if x2 > img1.shape[1]:
                    x2 = img1.shape[1]
                if y1 < 0:
                    y1 = 0
                if x1 < 0:
                    x1 = 0
                # check validity
                if y2-y1 < 1 or x2-x1 < 1:
                    y1 = x1 = 0
                    y2, x2 = img1.shape[:2]

                img1 = img1[y1:y2, x1:x2, :]
                img2 = img2[y1:y2, x1:x2, :]
            elif self.crop and not self.is_bbox:
                x1, y1, x2, y2 = babybox(self.yolo, img1, self.device)
                img1 = img1[y1:y2, x1:x2, :]
                img2 = img2[y1:y2, x1:x2, :]

        # Downsample image
        try:
            img1 = cv2.resize(img1, (self.H, self.W), interpolation=cv2.INTER_CUBIC)
            img2 = cv2.resize(img2, (self.H, self.W), interpolation=cv2.INTER_CUBIC)
        except:
            print('\n--------- ERROR! -----------\nUsual cv empty error')
            print(f'Shape of img1: {img1.shape}; Shape of im2: {img2.shape}')
            print(f'bbox: {bbox}')
            print(f'This is at idx: {idx}')
            exit(666)

        if self.augment:
            img1 = ToPILImage()(img1)
            img2 = ToPILImage()(img2)
            if self.flip_p > 0.5:
                img1 = TF.vflip(img1)
                img2 = TF.vflip(img2)
            if self.flip_p > 0.5:
                img1 = TF.hflip(img1)
                img2 = TF.hflip(img2)

            # img1 = TF.rotate(img1, self.rot)
            # img2 = TF.rotate(img2, self.rot)

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
        M = tr.div(img2 - img1, img1 + img2 + 1)         # +1 for numerical stability
        # M = tr.sub(M, tr.mean(M, (1, 2)).view(3, 1, 1))  # spatial intensity norm for each channel

        A = img1/255.  # convert image to [0, 1]
        A = tr.sub(A, tr.mean(A, (1, 2)).view(3, 1, 1))  # spatial intensity norm for each channel

        sample = ((A, M), target)

        # Video shape: C x D x H X W
        return sample
