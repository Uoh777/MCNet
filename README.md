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

- [VT5000](https://arxiv.org/pdf/2007.03262.pdf)
- [VT1000](https://arxiv.org/pdf/1905.06741.pdf)
- [VT821](https://link.springer.com/chapter/10.1007/978-981-13-1702-6_36)


## The Proposed Dataset
Our proposed RGBT SOD dataset VT723 that contain common challenging scenes of real world.
- VT723 [Google]()|[Baidu code:yuno](https://pan.baidu.com/s/1F171033a7JurP8ICq6Fv1w)



## Training & Testing & Evaluate
- Split the ground truth into skeleton map and contour map, which will be saved into `data/VT5000/skeleton` and `data/VT5000/contour`.
```shell
    python3 utils.py
```

- Train the model and get the pretrained model, which will be saved into `res` folder.
```shell
    python3 train.py
```

 - If you just want to evaluate the performance of MCNet without training, please download the [pretrained model]() into `res` folder
 - Train the model and get the predicted saliency maps, which will be saved into `eval/maps` folder.
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
- trained model: [Google]()|[Baidu code:p2t9](https://pan.baidu.com/s/1510TnLFx0gRK6S1ppXXSXg)


