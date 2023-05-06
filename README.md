# Open Parking Spot Detection
### DSBA 6156 - Applied Machine Learning
Parking spot detection is not new in practice, but a popular dataset that is used only has sunny, cloudy, and rain conditions.
The purpose of this project was to look at how well a model trained on the PKLot dataset performed on real world examples, as well as train a model
that was able to detect parking spots in poor weather conditions, like snow. To accomplish this our team needed a few essential components:
1. A dataset
2. A pretrained model
3. Augmentations
4. Additional images for training/testing
5. A way to label these new images

## Table of contents
- [General info](#general-info)
- [Technologies used](#technologies-used)
- [Process](#process)
  - [Image collection](#image-collection)
  - [Augmentations](#augmentations)
  - [Annotations](#annotations)
  - [Training](#training)
- [Results](#results)

## General info
![Image of failed parking spot detection](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/issue1.png)

## Technologies used
- Google Colab
- Python
- Yolo V7
- Label Studio

## Process
### Image collection

### Augmentations
Images were passed through an augmentation pipeline that added snow and rain effects to the original images.
By doing this, the model was able to learn on new weather conditions that it had not previously seen. This method was chose since finding
security camera footage of parking lots that also had views of enough parking spots was difficult to find. While there were a few livestreams and videos on Youtube
that our team was able to scrape frames from using a script, the augmentation process was faster as a proof of concept.

   Pre augmentation         |  Snow augmentation         |  Rain augmentation
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/normal1.png)  | ![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/snowaugment1.png)  |  ![](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/rainaugment1.png)


### Annotations
![Label Studio process](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/labelstudio.png)

### Training
```python
!python train.py --weights "/content/drive/MyDrive/6156/check_freeze/check.pt" --data "/content/data.yaml" --workers 4 --batch-size {batch_size} --img 640 --cfg cfg/training/yolov7.yaml --name yolov7 --epochs {epochs} --hyp data/hyp.scratch.p5.yaml --freeze 50
```

## Results
![Image of sunny parking spot detection](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/sunny1.png)
![Image of snow parking spot detection](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/snow1.png)
![Image of snow parking spot detection](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/snow2.png)
![Image of snow parking spot detection](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/snow3.png)
![Image of snow parking spot detection](https://github.com/pruthvirajcyn/Parking-spot-detection/blob/main/images/snow4.png)
