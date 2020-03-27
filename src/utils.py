import numpy as np
import cv2
import torch.nn.functional as F


def img2uint8(img):
    # convert to 8 bit if needed
    if img.dtype is np.dtype(np.uint16):
        if np.max(img[:]) < 256:
            scale = 255.  # 8 bit stored as 16 bit...
        elif np.max(img[:]) < 4096:
            scale = 4095.  # 12 bit
        else:
            scale = 65535.  # 16 bit
        img = cv2.convertScaleAbs(img, alpha=(225. / scale))
    return img


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


class ReferenceProcessor:
    """
    Reference pre-processor for DeepPhys architecture.
    Derivates and normalizes the reference signal.
    """

    def __init__(self, signal):
        self.signal = signal.astype(np.float)
        self.n = signal.size-1
        print(f"The length of training label: {self.n}")
        self.training_label = np.empty(shape=(self.n,), dtype=np.float)

    def calculate(self):
        self.__derivative()
        self.__scale()

    def __derivative(self):
        print("Derivating the signal...")
        for i in range(self.n):
            self.training_label[i] = self.signal[i+1]-self.signal[i]

    def __scale(self):
        print("Scaling the signal...")

        part = 0
        window = 32

        while part < (len(self.training_label) // window) - 1:
            self.training_label[part*window:(part+1)*window] /= np.std(self.training_label[part*window:(part+1)*window])
            part += 1

        if len(self.training_label) % window != 0:
            self.training_label[part * window:] /= np.std(self.training_label[part * window:])
