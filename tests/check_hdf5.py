import numpy as np
import h5py
import cv2


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

