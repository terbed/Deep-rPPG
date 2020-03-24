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


## GPU: Nvidia Jetson Nano
```
YOLO network inference time ===================================
Shape of the input network: torch.Size([1, 3, 416, 416])
The average running time of the network: 422.00370776938655 +/- 1.5917445608545957 ms



DeepPhys inference time =========================================
Shape of the input network: torch.Size([128, 3, 36, 36]) x 2
The average running time of the network: 202.41548820521507 +/- 1.2325808370819287 ms



PhysNet inference time ==========================================
Shape of the input network: torch.Size([1, 3, 128, 128, 128])
The average running time of the network: 2273.6186280514326 +/- 5.2405210041877925 ms


RateProbEst inference time ======================================
Shape of the input network: torch.Size([1, 1, 128])
The average running time of the network: 11.530726820404734 +/- 0.30963280228178724 ms


Full fused rate estimator: PhysNet+RateProbEst ===================
Shape of the input network: torch.Size([1, 3, 128, 128, 128])
The average running time of the network: 2274.0024768717817 +/- 7.459553790495938 ms


```