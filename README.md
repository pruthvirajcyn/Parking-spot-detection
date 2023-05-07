# Open Parking Spot Detection
### DSBA 6156 - Applied Machine Learning

## Table of contents
- [General info](#general-info)
- [Dataset](#dataset)
- [Process](#process)
  - [Image collection](#image-collection)
  - [Annotations](#annotations)
  - [Augmentations](#augmentations)
  - [Training](#training)
- [Technologies used](#technologies-used)
- [Pre-augmented model issues](#pre-augmented-model-issues)
- [Results](#results)
  - [Metrics](#metrics)
  - [Inferenced images](#inferenced-images)
- [Conclusion](#conclusion)

## General info
Parking spot detection is not new in practice, but a popular dataset that is used only has sunny, cloudy, and rain conditions.
The purpose of this project was to look at how well a model trained on the PKLot dataset performed on real world examples, as well as train a model
that was able to detect parking spots in poor weather conditions, like snow. To accomplish this our team needed a few essential components:
1. A dataset
2. A pretrained model
3. Augmentations
4. Additional images for training/testing
5. A way to label these new images

## Dataset
Here we talk about the PKLot dataset

## Technologies used
- <a href="https://colab.research.google.com/">Google Colab</a>
- <a href="https://www.python.org/">Python</a>
- <a href="https://github.com/WongKinYiu/yolov7">Yolo V7</a>
- <a href="https://labelstud.io/">Label Studio</a>

## Process
### Image collection
Additional images were collected through manually image search using common sources like Google, Pinterest, and other large image databases, and by using <a href="https://abhitronix.github.io/vidgear/v0.3.0-stable/">VidGear</a>. Vidgear is a high-performance video processing Python library that provide different "Gears" that are made for different video functionality. The CamGear "Gear" was used due to its ability to capture not only uploaded videos to Youtube, but also live streams such as the <a href="https://www.youtube.com/watch?v=c38K8IsYnB0">SUU Parking Lot Construction Camera</a> which was essential to capture different frames of snow covered parking lots.

### Annotations
Additional mages that were collected had to be annotated to provide the correct class for each parking spot, either space-empty or space-occupied, the same classes from the PKLot dataset.
This was accomplished by using <a href="https://labelstud.io/">Label Studio</a>, an open source data labeling tool. This provided a user friendly method of annotating
the ~150 additional images that were collected and a way to export the COCO formatted bounding box coordinates needed for the model.
<br><br>
![Label Studio process](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/labelstudio.png)

### Augmentations
Images were passed through an augmentation pipeline that added snow and rain effects to the original images. The <a href="https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library">Automold</a> library was chosen because it was designed to introduce real world scenarios for training neural networks of autonomous vehicles. This library provides many real world example augmentations, but our focus was only on snow and rain. The PKLot dataset as well as the non-snow images that were collected were passed through the augmentation pipeline, exponentially increasing the size of our dataset.

   Pre augmentation         |  Snow augmentation         |  Rain augmentation
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/normal1.png)  | ![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/snowaugment1.png)  |  ![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/rainaugment1.png)

### Training
Training was done using Google Colab, allowing each member of the group to have access to the model and weights, as well as access to a GPU. Initially, we attempted to train the YOLOv7 model directly on the PKLot dataset. However, we encountered some difficulties with this approach and found that the model was not learning effectively. To address this, we decided to freeze the first 50 layers of the YOLOv7 model during training. This approach helped to stabilize the training process and improve the overall performance of the model. Freezing the first 50 layers of the model sped up the training process significantly by allowing us to focus on training the later layers of the model, which are responsible for object detection. By using this modified YOLOv7 model, we were able to speed up the training of the model.
<br>
Since we were using Google Colab, we encountered issues with session timeouts, running out of GPU, and other cloud based problems. To combat this, the YOLOv7 train.py was modified to save the best.pt and last.pt weight files to a Google Drive so that progress was not lost. This also allowed our team to read from the Google Drive last.pt and best.pt files for retraining and inferencing. Batch size and number of epochs were set prior to each training session to maximize the use of the GPU.
<br><br>
The code below shows the cell used for training with the freeze flag set to 50.
```python
!python train.py --weights "/content/drive/MyDrive/6156/weights/last.pt" --data "/content/data.yaml" --workers 4 --batch-size {batch_size} --img 640 --cfg cfg/training/yolov7.yaml --name yolov7 --epochs {epochs} --hyp data/hyp.scratch.p5.yaml --freeze 50
```

## Pre-augmented model issues
The initial model trained on the PKLot dataset performed very well on the holdout dataset, but when used for inferencing with clear weather there were obvious problems. We believe that the limited camera angles, while great for this parking lot, was not representative of other parking lots that we were able to find. Our team believes that a camera mounted higher and with a smaller <a href="https://ipvm.com/reports/testing-camera-height">angle of incidence</a> provides a better view of a parking lot, thus able to determine which spots are empty and which are occupied. Unfortunately, not all locations are able to get cameras high enough to accomplish this, so additionaly training is needed to account for these angles.

   Camera 1         |  Camera 2         |  Camera 3
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/2012-09-11_15_38_53_jpg.rf.bcdabdb175b85ebb981248ddc666e1d7.jpg)  | ![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/2012-12-21_17_25_13_jpg.rf.f1800300d28b08e400977e1b74594f88.jpg)  |  ![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/2013-03-05_09_45_04_jpg.rf.e8589d4a1ec587d59d96187a21b6568e.jpg)


Below is an image that was inferenced using a model trained just on the PKLot dataset. The model performed inadequately in detecting most parking spots in the image and even mistakenly identified trees as parking spots. The problem was not limited to this parking lot, but it serves as an example of how even though the model performed well on its test set it may not work well if deployed.
<br><br>
   PKLot dataset         
:-------------------------:
![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/pklot_matrix.png) |
![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/pklot_f1.png)

## Results
### Metrics
   Snow augmented Dataset         
:-------------------------:
![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/snow_matrix.png) |
![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/snow_f1.png)

   Rain augmented dataset         
:-------------------------:
![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/rain_matrix.png) |
![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/rain_f1.png)

### Inferenced images
![Image of sunny parking spot detection](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/sunny1.png)
![Image of snow parking spot detection](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/snow1.png)
![Image of snow parking spot detection](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/snow2.png)
![Image of snow parking spot detection](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/snow3.png)
![Image of snow parking spot detection](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/snow4.png)

## Conclusion
Overall the model trained on images augmented with snow and rain, additional images that captured additional security camera angles, and real world snow covered parking lots, performed well. There are still improvements that could be made, but a dataset limited to three camera angles and the difficulty in finding additional images make that a challenge.
