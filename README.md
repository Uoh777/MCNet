## MCNet
MCNet: Mirror Complementary Network for RGB-thermal Salient Object Detection


## Prerequisites
- [Python 3.68](https://www.python.org/)
- [Pytorch 1.3.1](http://pytorch.org/)
- [Cuda 10.0](https://developer.nvidia.com/cuda-10.0-download-archive)
- [OpenCV 4.1.2](https://opencv.org/)
- [Numpy 1.17.3](https://numpy.org/)
- [TensorboardX 2.1](https://github.com/lanpa/tensorboardX)


## Benchmark Datasets
Download the following datasets and unzip them into `data` folder

- [VT5000](https://drive.google.com/file/d/1q3cgxs3_go4yO1iB2zLNEhZXdN60Mdap/view?usp=sharing)
- [VT1000](https://drive.google.com/file/d/1I4GPXOl-xQPi7SSHx5NqgqFqtVm9XbHO/view?usp=sharing)
- [VT821](https://drive.google.com/file/d/1hXJWFE2sSs0mIsm1ygMDoLpL3OJ8Eiz-/view?usp=sharing)


## The Proposed Dataset
Our proposed RGBT SOD dataset VT723 that contain common challenging scenes of real world.
- VT723 [Google](https://drive.google.com/file/d/1Vi91vzn7xym238wu5QUh7URRi7jn4Vru/view?usp=sharing)|[Baidu code:yuno](https://pan.baidu.com/s/1F171033a7JurP8ICq6Fv1w)



## Training & Testing & Evaluate
- Split the ground truth into skeleton map and contour map, which will be saved into `data/VT5000/skeleton` and `data/VT5000/contour`.
```shell
    python3 utils.py
```

- Train the model and get the pretrained model, which will be saved into `res` folder.
```shell
    python3 train.py
```

 - If you just want to evaluate the performance of MCNet without training, please download the [pretrained model](https://drive.google.com/file/d/1qcZeBiwF78Lv24hXmXN4vMFbK4yC-C-y/view?usp=sharing) into `res` folder.
 - Test the model and get the predicted saliency maps, which will be saved into `maps` folder.
 ```shell
    python3 test.py
```

- Evaluate the predicted results. 
```shell
    cd eval
    matlab
    main
```


## Saliency maps & Trained model
- saliency maps: [Google]()|[Baidu code:p2t9](https://pan.baidu.com/s/1510TnLFx0gRK6S1ppXXSXg)
- trained model: [Google](https://drive.google.com/file/d/1qcZeBiwF78Lv24hXmXN4vMFbK4yC-C-y/view?usp=sharing)|[Baidu code:p2t9](https://pan.baidu.com/s/1510TnLFx0gRK6S1ppXXSXg)


