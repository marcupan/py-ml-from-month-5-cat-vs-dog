# Cat vs. Dog Image Classifier

## Project Overview
This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch to classify images of cats and dogs. The goal is to achieve an accuracy of over 90% on the classification task.

## Learning Objectives
- Understand and implement CNNs using PyTorch
- Learn how to process and augment image data
- Gain experience with training deep learning models
- Practice model evaluation and saving techniques

## Dataset
This project uses the popular Cats vs. Dogs dataset, which contains thousands of labeled images of cats and dogs. The dataset is organized in the following structure:

```
data/
├── raw/
│   └── cats-vs-dogs/
│       ├── train/
│       │   ├── cats/
│       │   └── dogs/
│       └── test/
│           ├── cats/
│           └── dogs/
```

If the dataset is not found, the code will fall back to using CIFAR-10 as a substitute for demonstration purposes.

## Project Structure
- `requirements.txt`: List of required Python packages
- `data_loader.py`: Functions for loading and preprocessing the dataset
- `model.py`: Definition of the CNN architecture
- `train.py`: Script for training the model
- `utils.py`: Utility functions for the project
- `evaluate.py`: Script for evaluating the trained model

## Requirements
- Python 3.7+
- PyTorch 1.7+
- torchvision
- Pillow
- numpy
- matplotlib

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Train the model:
```bash
python train.py
```

2. Evaluate the model:
```bash
python evaluate.py
```

## Model Architecture
The CNN architecture consists of:
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for classification
- Dropout for regularization

## Results
The model aims to achieve >90% accuracy on the test set.

## References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Cats vs. Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats)
- [CNN Explanation](https://cs231n.github.io/convolutional-networks/)
