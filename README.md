# Food Vision with EfficientNet

This repository contains the code for the Food Vision project using EfficientNet. The project involves building a deep learning model to classify food images into 101 different classes using the Food101 dataset.


## Project Overview

This project utilizes TensorFlow and EfficientNet for image classification. It involves training a model on the Food101 dataset, fine-tuning the model, and evaluating its performance.

## Dataset

The [Food101 dataset](https://www.tensorflow.org/datasets/catalog/food101) is used for this project. It consists of 101,000 images across 101 food classes.

## Data Preprocessing

The dataset is preprocessed using TensorFlow Datasets (TFDS). Images are resized, normalized, and batched to create an efficient input pipeline for the model.

## Model Architecture

The EfficientNetV2B0 architecture is used as the base model for feature extraction. The top layers are added for classification. The model is compiled with a suitable loss function, optimizer, and metrics.

## Training

The model is trained on the preprocessed data, and the training process is logged using TensorBoard. Checkpoints are saved to monitor the model's progress.

## Fine-tuning

After feature extraction, the model is fine-tuned on the entire Food101 dataset. Learning rate reduction and early stopping callbacks are used to optimize training.

## Results

The model's performance is evaluated on the test set, and the results are compared before and after fine-tuning.
