# 102 Flower Classification using ResNet50 and Transfer Learning

This repository contains the code for the **102 Flower Classification** task, part of my project for **Fellowship.AI**. The aim is to classify flower images into 102 different categories using **ResNet50** architecture and transfer learning techniques.

## Project Overview

This project leverages **ResNet50**, a pre-trained deep learning model, to classify flower images. The model is fine-tuned to recognize 102 different flower species using transfer learning, where the pre-trained weights are adapted to our specific dataset.

### Dataset

The dataset consists of images of 102 flower species. 

download data from: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

The dataset includes:

- Training file (trnid.mat): Contains the image IDs used for training the model.
- Validation file (valid.mat): Contains the image IDs used for validating the model's performance.
- Test file (tstid.mat): Contains the image IDs used for testing the model after training.
- Labels file (imagelabels.mat): Links each image to a specific flower species (1-102).
- Images: The flower images are provided in .jpg format, with file names like image_XXXX.jpg.

### Model: ResNet50

**ResNet50** is a widely adopted residual neural network architecture for image classification tasks. It consists of 50 layers and uses residual connections to help train deep models more effectively.

In this project, we took the following steps:

1. **Loading Pre-trained ResNet50**: We started with a pre-trained ResNet50 model, which was trained on the ImageNet dataset.
2. **Freezing Early Layers**: The initial layers were frozen to keep the pre-trained weights and retain general image features.
3. **Customizing Output Layer**: The final fully connected layer was modified to fit the flower classification task, adjusting the output to 102 classes.
4. **Training**: The model was fine-tuned using the flower dataset to optimize its performance for the specific task.


## Requirements

- Python 3.x
- PyTorch
- torchvision
- pandas
- numpy
- matplotlib

## Instructions for Running

1. Clone this repository:
   ```bash
   git clone https://github.com/kavya10/flower_102.git

## References

1. https://github.com/lukysummer/Transfer-Learning-with-ResNet50/blob/main/Transfer_Learning_ResNet50_Part1.ipynb
