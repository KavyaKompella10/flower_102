# 102 Flower Classification using ResNet50 and Transfer Learning

This repository contains the code for the **102 Flower Classification** task, part of my project for **Fellowship.AI**. The aim is to classify flower images into 102 different categories using **ResNet50** architecture and transfer learning techniques.

## Project Overview

This project leverages **ResNet50**, a pre-trained deep learning model, to classify flower images. The model is fine-tuned to recognize 102 different flower species using transfer learning, where the pre-trained weights are adapted to our specific dataset.

### Dataset

The dataset consists of images of 102 flower species. It is divided into two parts:

- **Training Data**: Located in the `trnid.csv` file, containing the IDs of images used for training.
- **Labels**: Located in the `labels.csv` file, which maps each image ID to a corresponding label. The image files are stored in the `jpg` folder, where each image is named in the format `image_00001.jpg`.

Additionally, we have a **validation dataset** with a similar structure to evaluate model performance.

### Model: ResNet50

**ResNet50** is a widely adopted residual neural network architecture for image classification tasks. It consists of 50 layers and uses residual connections to help train deep models more effectively.

In this project, we took the following steps:

1. **Loading Pre-trained ResNet50**: We started with a pre-trained ResNet50 model, which was trained on the ImageNet dataset.
2. **Freezing Early Layers**: The initial layers were frozen to keep the pre-trained weights and retain general image features.
3. **Customizing Output Layer**: The final fully connected layer was modified to fit the flower classification task, adjusting the output to 102 classes.
4. **Training**: The model was fine-tuned using the flower dataset to optimize its performance for the specific task.

## Repository Structure

### Files

- **`trnid.csv`**: Contains the image IDs used for training.
- **`labels.csv`**: Contains the labels for each image ID, corresponding to one of the 102 flower categories.
- **`jpg/`**: This folder contains the images in `.jpg` format, each named `image_XXXXX.jpg`.
- **`valid_data/`**: Contains the validation dataset, similar in structure to the training data.

### Notebooks

- **`file_directory.ipynb`**: Organizes and processes the image files, ensuring proper format for training and validation.
- **`mat2csv.ipynb`**: Converts data from `.mat` to `.csv` format for easier integration into the training pipeline.
- **`model.ipynb`**: Contains the core code for building, training, and evaluating the ResNet50 model for flower classification.

## Process

1. **Data Preprocessing**: Images are loaded and resized to match the input size required by ResNet50 (224x224 pixels).
2. **Model Fine-tuning**: The pre-trained ResNet50 is fine-tuned with new layers for flower classification, training the model on the dataset.
3. **Evaluation**: The model is evaluated on the validation dataset to assess accuracy in classifying different flower species.

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
   git clone https://github.com/your-username/flower-classification.git

## References

1. "https://github.com/lukysummer/Transfer-Learning-with-ResNet50/blob/main/Transfer_Learning_ResNet50_Part1.ipynb"
