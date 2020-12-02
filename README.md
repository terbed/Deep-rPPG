# Deep-rPPG: Camera-based pulse estimation using deep learning tools
Deep learning (neural network) based remote photoplethysmography: how to extract pulse signal from video using deep learning tools
**Source code of the master thesis titled "Camera-based pulse estimation using deep learning tools"**

## Implemented networks
### DeepPhys
Chen, Weixuan, and Daniel McDuff. "Deepphys: Video-based physiological measurement using convolutional attention networks." Proceedings of the European Conference on Computer Vision (ECCV). 2018.

### PhysNet
Yu, Zitong, Xiaobai Li, and Guoying Zhao. "Remote photoplethysmograph signal measurement from facial videos using spatio-temporal networks." Proc. BMVC. 2019.

## NVIDIA Jetson Nano inference
The running speed of the networks are tested on NVIDIA Jetson Nano. Results and the installation steps of `PyTorch` and `OpenCV` are in the `nano` folder.


## Thesis will be uploaded soon!
# Abstract of the corresponding master thesis
## titled "Camera-based pulse estimation using deep learning tools"
Lately, it has been shown that an average color camera is able to detect the subtle color variations of the skin (caused by the cardiac cycle) – enabling us to monitor the pulse remotely in a non-contact manner, with a camera. Since then, the field of remote photoplethysmography (rPPG) has been formed and advanced quickly in order the overcome its main limitations, namely: motion robustness and weak signal quality. Most recently, deep learning (DL) methods have also appeared in the field – applied only on adults. In this work we utilize DL approaches for long-term, continuous premature infant monitoring in the Neonatal Intensive Care Unit (NICU). 

The technology used in NICU for monitoring vital signs of the infants has hardly changed in the past 30 years (i.e. ECG and pulse-oximetry). Despite the fact that these technologies has been of great importance for the reliable measurement of essential vital signs (like heart-rate, respiration-rate and blood oxygenation), they also have considerable disadvantages – originating from their contact nature. The skin of premature infants are fragile and contact sensors may cause discomfort, stress, pain and even injuries – thus can have a negative impact on the early development of the neonate. For the well-being of not exclusively newborns, but also every patient or subject who requires long-term monitoring (e.g. elders) or for whom contact sensors are not applicable (e.g. burn patients), it would be beneficial to replace contact-based technologies with non-contact alternatives without significantly sacrificing accuracy. Therefore, the topic of this study is camera-based (remote) pulse monitoring -- utilizing DL methods -- in the specific use-case of infant monitoring in the NICU.

First of all, as there is no publicly available infant database for rPPG purposes currently to our knowledge, it had to be collected for Deep Neural Network (DNN) training and evaluation. Video data from infants was collected in the $I$st Dept. of Neonatology of Pediatrics, Dept. of Obstetrics and Gynecology, Semmelweis University, Budapest, Hungary and a database was created for DNN training and evaluation with a total length of around 1 day. 

Two state-of-the-art DNNs were implemented (and trained on our data) which was specifically developed for the task of pulse extraction from video, namely 
DeepPhys and PhysNet. In addition, two classical algorithms were implemented, namely POS and FVP, to be able to compare the two approach: in our dataset DL methods outperform classical ones. A novel data augmentation technique is introduced for rPPG DNN training, namely frequency augmentation, which is essentially a temporal resampling of a video and corresponding label segment (while keeping the original camera sampling rate parameter unchanged) resulting in a modified pulse-rate. This method significantly improved the generalization capability of the DNNs.

In case of some external condition the efficacy of remote sensing the vital signs are degraded (e.g. inadequate illumination, heavy subject motion, limited visible skin surface etc..). In these situations the prediction of the methods might be inaccurate or might give a completely wrong estimate blindly without warning -- which is undesirable especially in medical applications. To solve this problem the technique of Stochastic Neural Networks (SNNs) is proposed which yields a probability distribution over the whole output space instead of a single point estimate. In other words, SNNs associate a certainty/confidence/quality measure to their prediction, therefore we know how reliable an estimate is. In the spirit of this a probabilistic neural network was designed for pulse-rate estimation, called RateProbEst, fused and trained together with PhysNet. This method has not been applied in this field before to our knowledge. Each method was evaluated and compared with each other on a large benchmark dataset.

Finally, the feasibility of rPPG DNN applications in resource limited environment is inspected on a NVIDIA Jetson Nano embedded system. The results demonstrate that the implemented DNNs are capable of (quasi) real-time inference even on a limited hardware.


## Special application on neonates
A custom YOLO network is used to crop the baby as a preprocessing step.
This network was created based on this repo: [https://github.com/eriklindernoren/PyTorch-YOLOv3][PyTorch-YOLOv3]

Our modified version:
https://github.com/terbed/PyTorch-YOLOv3

[PyTorch-YOLOv3]: https://github.com/eriklindernoren/PyTorch-YOLOv3
