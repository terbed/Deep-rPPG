import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import h5py

from yolo.detect import babybox
from yolo.models import *
from yolo.utils import *
tr = torch


def load_input(img_path, device):
    img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
    # Resize
    img = F.interpolate(img.unsqueeze(0), size=416, mode="nearest")
    img = img.type(tr.FloatTensor)
    img = img.to(device)

    return img


def load_input_hdf5(path, idx, device):
    with h5py.File(path, 'r') as db:
        frames = db['frames']
        print(f'Video shape: {frames.shape}')
        img = frames[idx, :, :, :]

        print(f'image shape: {img.shape}')
        inp = transforms.ToTensor()(img)

        # img = cv2.cvtColor(frames[idx, :, :, :].squeeze(), cv2.COLOR_RGB2BGR)

        inp = F.interpolate(inp.unsqueeze(0), size=416, mode="nearest")
        inp = inp.type(tr.FloatTensor)
        inp = inp.to(device)
        print(f'Network input: {inp.shape}')

        return img, inp


device = tr.device('cuda') if tr.cuda.is_available() else tr.device('cpu')

model_def = 'yolo/config/yolov3-custom.cfg'
class_path = 'yolo/config/classes.names'
weight_path = 'yolo/weights/yolov3_ckpt_42.pth'

# root_hdf5 ='/media/nas/PUBLIC/benchmark_set/'
# hdf5_path = "test_series_9_febr25.hdf5"
# hdf5_path = 'breathandpulsebenchmark_128x128_8UC3_minden.hdf5'

root_hdf5 = '/media/nas/PUBLIC/0_training_set/'
hdf5_path = "PIC190111_128x128_8UC3.hdf5"

# img_path = '/media/terbe/sztaki/DATA/BabyCropper/data/test_baby/'
# img_name = '000028.png'
#
# img_path = '/media/terbe/sztaki/DATA/BabyCropper/data/images128/'
# img_name = '2020y2m16d_9h32m_001347.png'

# -----------------
# parameters
# -----------------
conf_thres = 0.8
nms_thres = 0.4

# --------------------
# Load model
# -------------------
model = Darknet(model_def).to(device)
model.load_state_dict(torch.load(weight_path))
model.eval()

# Load classes
classes = load_classes(class_path)  # Extracts class labels from file

# Load image
# inp = load_input(img_path+img_name, device)
# img = cv2.imread(img_path+img_name)

for i in range(1, 128):
    img, inp = load_input_hdf5(root_hdf5+hdf5_path, i, device)

    x_1, y_1, x_2, y_2 = babybox(model, img, device)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 500, 500)
    img = cv2.rectangle(img, (x_1, y_1), (x_2, y_2), (0, 0, 0), 1)
    cv2.imshow('frame', img)

    cv2.namedWindow('cropped', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cropped', 500, 500)
    cv2.imshow('cropped', img[y_1:y_2, x_1:x_2, :])
    cv2.waitKey(1)

while cv2.waitKey(1) != 13:
    pass
cv2.destroyAllWindows()
