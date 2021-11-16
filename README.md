## MCNet
MCNet: Mirror Complementary Network for RGB-thermal Salient Object Detection


## Prerequisites
- [Python 3.68](https://www.python.org/)
- [Pytorch 1.3.1](http://pytorch.org/)
- [Cuda 10.0](https://developer.nvidia.com/cuda-10.0-download-archive)
- [OpenCV 4.1.2](https://opencv.org/)
- [Numpy 1.17.3](https://numpy.org/)
- [TensorboardX 2.1](https://github.com/lanpa/tensorboardX)


## Download dataset
Download the following datasets and unzip them into `data` folder

- [VT5000](https://arxiv.org/pdf/2007.03262.pdf)
- [VT1000](https://arxiv.org/pdf/1905.06741.pdf)
- [VT821](https://link.springer.com/chapter/10.1007/978-981-13-1702-6_36)
- [VT723]([Baidu: https://pan.baidu.com/s/1F171033a7JurP8ICq6Fv1w] [code: yuno])


## Training & Testing

- Train the MCNet:

    `python train.py `

- Test the MCNet:

    `python test.py `
    
    The test maps will be saved to './maps/'.

- Evaluate the result maps:

	eval/main.m


## Trained model
- trained model: ([Baidu: https://pan.baidu.com/s/1510TnLFx0gRK6S1ppXXSXg] [code: p2t9])


