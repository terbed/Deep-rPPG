# Deep-rPPG
Deep learning (neural network) based remote photoplethysmography: how to extract pulse signal from video using deep learning tools

## Implemented networks
### DeepPhys
Chen, Weixuan, and Daniel McDuff. "Deepphys: Video-based physiological measurement using convolutional attention networks." Proceedings of the European Conference on Computer Vision (ECCV). 2018.

### PhysNet
Yu, Zitong, Xiaobai Li, and Guoying Zhao. "Remote photoplethysmograph signal measurement from facial videos using spatio-temporal networks." Proc. BMVC. 2019.

## Special application on neonates
A custom YOLO network is used to crop the baby as a preprocessing step.
This network was created based on this repo: [https://github.com/eriklindernoren/PyTorch-YOLOv3][PyTorch-YOLOv3]

Our modified version:
https://github.com/terbed/PyTorch-YOLOv3

[PyTorch-YOLOv3]: https://github.com/eriklindernoren/PyTorch-YOLOv3

## NVIDIA Jetson Nano inference
The running speed of the networks are tested on NVIDIA Jetson Nano. Results and the installation steps of `PyTorch` and `OpenCV` are in the `nano` folder.
