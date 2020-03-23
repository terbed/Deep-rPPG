import numpy as np
import h5py
import cv2
import torch
tr = torch


def check_bbox():
    path = '../../DATA/PIC190111_128x128_8UC3_copy.hdf5'
    with h5py.File(path, 'r') as db:
        N = db['frames'].shape[0]
        print(db['bbox'].shape)
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('video', 500, 500)
    for i in range(3000, N):
        with h5py.File(path, 'r') as db:
            img = db['frames'][i, :]
            bbox = db['bbox'][i, :]

        print(bbox)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.rectangle(img, (bbox[2], bbox[0]), (bbox[3], bbox[1]), (255, 0, 0), 2)

        cv2.imshow('video', img)
        cv2.waitKey(5)
    cv2.destroyAllWindows()


def check_fast_dataset():
    from src.dset import Dataset4DFromHDF5
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    my_dset = Dataset4DFromHDF5('/Volumes/sztaki/DATA/PIC191111_128x128_U8C3_fast.hdf5',
                                ('PulseNumerical',), device=device, crop=False, augment_freq=True)

    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('video', 500, 500)
    n = 7
    for i in range(n):
        video, pr = my_dset[i]
        print(pr)
        video = video[0].data.cpu().permute(1, 2, 3, 0).numpy()
        for j in range(video.shape[0]):
            frame = np.squeeze(video[j, :])
            frame = frame - np.min(frame)
            frame = frame / np.max(frame)
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow('video', frame)
            cv2.waitKey(40)

    cv2.destroyAllWindows()

# check_bbox()
check_fast_dataset()