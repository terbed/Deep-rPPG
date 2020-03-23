from src.dset import *
from torch.utils.data import DataLoader

def convert2cvimshow(img):
    frame = img.copy()
    frame = frame - np.min(frame)
    frame = frame / np.max(frame)
    frame = (frame * 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame


def physnet_dset_test(idx):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # my_dset = Dataset4DFromHDF5('/media/nas/PUBLIC/benchmark_set/breathandpulsebenchmark_128x128_8UC3_minden.hdf5',
    #                            ('PulseNumerical', ), device=device)

    my_dset = Dataset4DFromHDF5('/Volumes/sztaki/DATA/PIC191111_128x128_U8C3_fast.hdf5',
                                ('PulseNumerical',), device=device, crop=False, augment_freq=True)

    dloader = DataLoader(my_dset, 2, True, pin_memory=True)

    it = iter(dloader)
    video, label = next(it)
    print(video[0].type)
    print(label.shape)
    print(label*60.)

    label = label.data.cpu().numpy()

    # Display image
    # C x D x H X W
    video = video[0].data.cpu()[0, :].permute(1, 2, 3, 0).numpy()
    # print(video.shape)

    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('video', 500, 500)
    for i in range(video.shape[0]):
        frame = np.squeeze(video[i, :])
        frame = frame - np.min(frame)
        frame = frame/np.max(frame)
        frame = (frame*255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('video', frame)
        cv2.waitKey(40)

    cv2.destroyAllWindows()
    # plot label
    # print(label.shape)
    # plt.figure(figsize=(12, 6))
    # plt.plot(label)
    # plt.grid()
    # plt.show()


def deepphys_dset_test(idx):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dset = DatasetDeepPhysHDF5('/media/nas/PUBLIC/0_training_set/PIC190111_128x128_8UC3_fast.hdf5', device)
    A, M, point = dset[idx]

    print('target: ', point)

    A = A.permute(1, 2, 0).data.cpu().numpy()
    M = M.permute(1, 2, 0).data.cpu().numpy()

    cv2.namedWindow('A', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('A', 500, 500)
    A = convert2cvimshow(A)
    cv2.imshow('A', A)

    cv2.namedWindow('M', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('M', 500, 500)
    M = convert2cvimshow(M)
    cv2.imshow('M', M)

    while cv2.waitKey(1) != 13:
        pass


if __name__ == "__main__":
    physnet_dset_test(8000)                                                  # OK
    # deepphys_dset_test(10)                                                     # OK
