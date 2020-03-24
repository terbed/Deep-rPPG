import sys
sys.path.append('../')

from src.archs import *
from yolo.models import *
from yolo.utils import *
import time
import torch

tr = torch
tr.backends.cudnn.benchmark = True
tr.backends.cudnn.deterministic = True
cuda = tr.cuda.is_available()


def sync():
    if cuda:
        tr.cuda.synchronize()


def timer(func):
    total_runs = 50    # repeat number of inference running
    warm_up = 10        # warm up iterations for GPU

    def wrapper(*args, **kwargs):
        durations = []
        for i in range(total_runs):
            sync()
            start = time.perf_counter()
            func(*args, **kwargs)
            sync()
            duration = (time.perf_counter() - start) * 1000.  # duration in msec
            print(f'{i}th repetition duration: {duration} ms')
            if i > warm_up:
                durations.append(duration)

        avg_dur = np.mean(durations)
        std = np.std(durations)
        print('\n----------------------------------------------------------------------------------')
        print(f'The average running time of the network: {avg_dur} +/- {std} ms')
        print('------------------------------------------------------------------------------------')

    return wrapper


@timer
def run_inference(network, *x):
    with tr.no_grad():
        network(*x)


def measure_yolo(*shape):
    model_def = '../yolo/config/yolov3-custom.cfg'
    # weight_path = 'yolo/weights/yolov3_ckpt_42.pth'
    yolo = Darknet(model_def).to(device)
    # yolo.load_state_dict(torch.load(weight_path))
    yolo.eval()
    print("\nYOLO network is initialized and ready to work!")
    yolo_input = tr.randn(*shape).to(device)
    print(f'Shape of the input network: {yolo_input.shape}')
    run_inference(yolo, yolo_input)


def measure_deepphys(*shape):
    print('\n\nDeepPhys inference time =========================================')
    deepphys = DeepPhys().to(device).eval()
    dp_input = (tr.randn(*shape).to(device), tr.randn(*shape).to(device))
    print(f'Shape of the input network: {dp_input[0].shape} x 2')
    run_inference(deepphys, *dp_input)


def measure_physnet(*shape):
    print('\n\nPhysNet inference time ==========================================')
    physnet = PhysNetED().to(device).eval()
    pn_input = tr.randn(*shape).to(device)
    print(f'Shape of the input network: {pn_input.shape}')
    run_inference(physnet, pn_input)


def measure_rateprobest(*shape):
    print('\n\nRateProbEst inference time ======================================')
    ratest = RateProbEst().to(device).eval()
    ratest_input = tr.randn(*shape).to(device)
    print(f'Shape of the input network: {ratest_input.shape}')
    run_inference(ratest, ratest_input)


def measure_fullestimator(*shape):
    print('\n\nFull fused rate estimator: PhysNet+RateProbEst ===================')
    ratest = RateProbEst().to(device).eval()
    physnet = PhysNetED().to(device).eval()
    pn_input = tr.randn(*shape).to(device)
    print(f'Shape of the input network: {pn_input.shape}')

    def fused_rate_estimator(x):
        ratest(physnet(x).view(-1, 1, 128))

    run_inference(fused_rate_estimator, pn_input)


if __name__ == '__main__':
    device = tr.device('cuda') if tr.cuda.is_available() else tr.device('cpu')
    print(device)

    # ---------------------
    # YOLO inference time
    # ---------------------
    measure_yolo(1, 3, 128, 128)

    # ----------------------------
    # DeepPhys inference
    # ----------------------------
    measure_deepphys(128, 3, 36, 36)

    # ------------------------------
    # PhysNet
    # ------------------------------
    measure_physnet(1, 3, 128, 128, 128)

    # ------------------------------
    # RateProbEst
    # ------------------------------
    measure_rateprobest(1, 1, 128)

    # --------------------------------
    # Fused Full Rate Estimator
    # ---------------------------------
    measure_fullestimator(1, 3, 128, 128, 128)


