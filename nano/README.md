# NVIDIA Jetson Nano
Test networks on jetson nano

## Installing pytorch and opencv
__(1) Creating virtual environment__ ---------------------------------------------------

```
sudo apt-get update && sudo apt-get upgrade

sudo apt-get install git cmake
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install libhdf5-serial-dev hdf5-tools
sudo apt-get install python3-dev

wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
rm get-pip.py

sudo pip install virtualenv virtualenvwrapper
```

Copy the following to .bashrc:
```
nano ~/.bashrc
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
```

```
source ~/.bashrc
mkvirtualenv torch -p python3
workon torch
pip install numpy
```

__(2) Link system opencv__ ---------------------------------------------------

OpenCV is built to the system by default, we have to link it to our environment.

Find opencv .so file: `find /usr/local -name "*opencv*" -o -name "*cv2*"`
Link the file:
```
cdsitepackages # enters current virtualenv's site-packages directory
ln -s /usr/local/lib/python3.6/dist-packages/cv2/python-3.6/cv2.cpython-36m-aarch64-linux-gnu.so cv2.cpython-36m-aarch64-linux-gnu.so
```

__(3) Install pytorch__ --------------------------------------------------------

Inside the env:
```
wget https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base
pip install Cython
pip install numpy torch-1.4.0-cp36-cp36m-linux_aarch64.whl
```

__(3) Install torchvision__ --------------------------------------------------------

Inside the env:
```
sudo apt-get install libjpeg-dev zlib1g-dev
git clone --branch v0.5.0 https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
cd torchvision
python setup.py install
cd ../

```

__(4) Verification__ --------------------------------------------------------
```
>>> import cv2
>>> print(cv2.__version__)

>>> import torch
>>> print(torch.__version__)
>>> print('CUDA available: ' + str(torch.cuda.is_available()))
>>> print('cuDNN version: ' + str(torch.backends.cudnn.version()))
>>> a = torch.cuda.FloatTensor(2).zero_()
>>> print('Tensor a = ' + str(a))
>>> b = torch.randn(2).cuda()
>>> print('Tensor b = ' + str(b))
>>> c = a + b
>>> print('Tensor c = ' + str(c))

>>> import torchvision
>>> print(torchvision.__version__)
```

# Results
50 repetition was conducted and the last 40 running time was averaged (warming up time).
## CPU: 2.6 GHz Quad-Core Intel Core i7 (2012)
```
YOLO network inference time ========================================================
Shape of the input network: torch.Size([1, 3, 128, 128])
The average running time of the network: 228.9040875897436 +/- 7.853026535619944 ms



DeepPhys inference time ==============================================================
Shape of the input network: torch.Size([128, 3, 36, 36]) x 2
The average running time of the network: 754.5646981794863 +/- 12.488300641005996 ms



PhysNet inference time =============================================================
Shape of the input network: torch.Size([1, 3, 128, 128, 128])
The average running time of the network: 10930.139147512828 +/- 1298.24171836462 ms



RateProbEst inference time ==========================================================
Shape of the input network: torch.Size([1, 1, 128])
The average running time of the network: 9.981026487177283 +/- 0.8496852697874339 ms



Full fused rate estimator: PhysNet+RateProbEst ========================================
Shape of the input network: torch.Size([1, 3, 128, 128, 128])
The average running time of the network: 10539.158413282054 +/- 1036.8366352881399 ms
```


## GPU: Nvidia Jetson Nano
```
YOLO network inference time =======================================================
Shape of the input network: torch.Size([1, 3, 128, 128])
The average running time of the network: 71.27102261517557 +/- 1.4345624959190526 ms

Shape of the input network: torch.Size([1, 3, 416, 416])
The average running time of the network: 428.9916333333307 +/- 1.3436190679117026 ms


DeepPhys inference time ============================================================
Shape of the input network: torch.Size([128, 3, 36, 36]) x 2
The average running time of the network: 186.80833720491836 +/- 1.3538288588565277 ms



PhysNet inference time ==============================================================
Shape of the input network: torch.Size([1, 3, 128, 128, 128])
The average running time of the network: 1877.2514318975132 +/- 6.865573963303903 ms



RateProbEst inference time ===========================================================
Shape of the input network: torch.Size([1, 1, 128])
The average running time of the network: 9.087809358998829 +/- 0.2438779403287001 ms

Shape of the input network: torch.Size([10, 1, 128])
The average running time of the network: 14.80706958980777 +/- 0.19120636422604642 ms


Full fused rate estimator: PhysNet+RateProbEst ========================================
Shape of the input network: torch.Size([1, 3, 128, 128, 128])
The average running time of the network: 1884.9364145128511 +/- 8.400773451781818 ms

Shape of the input network: torch.Size([3, 3, 128, 128, 128]) -> (max batch size: 3)
The average running time of the network: 5674.588254154001 +/- 37.78818835674967 ms

```