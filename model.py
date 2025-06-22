"""
CNN Model for Cat vs. Dog Image Classifier

This module defines the CNN architecture for classifying images of cats and dogs.
It includes a detailed explanation of each layer and its purpose.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CatDogCNN(nn.Module):
    """
    Convolutional Neural Network for cat vs. dog classification.

    Architecture:
    - 4 convolutional blocks (conv -> batch_norm -> relu -> pool)
    - 2 fully connected layers
    - Dropout for regularization
    """

    def __init__(self):
        """Initialize the network architecture."""
        super(CatDogCNN, self).__init__()

        # First convolutional block
        # Input: 3x224x224 (RGB image)
        # Output: 32x112x112
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        # Input: 32x112x112
        # Output: 64x56x56
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional block
        # Input: 64x56x56
        # Output: 128x28x28
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fourth convolutional block
        # Input: 128x28x28
        # Output: 256x14x14
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the flattened features
        # After 4 pooling layers with stride 2, the spatial dimensions are reduced by 2^4 = 16
        # 224/16 = 14, so the feature map size is 256x14x14
        self.fc_input_size = 256 * 14 * 14

        # First fully connected layer
        # Input: 256 * 14 * 14
        # Output: 512
        self.fc1 = nn.Linear(self.fc_input_size, 512)

        # Dropout layer for regularization (prevents overfitting)
        # Randomly zeroes some of the elements with probability p=0.5
        self.dropout = nn.Dropout(p=0.5)

        # Second fully connected layer (output layer)
        # Input: 512
        # Output: 2 (binary classification: cat or dog)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2)
        """
        # First convolutional block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Second convolutional block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Third convolutional block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Fourth convolutional block
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Flatten the output for the fully connected layer
        x = x.view(-1, self.fc_input_size)

        # First fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Apply dropout for regularization
        x = self.dropout(x)

        # Second fully connected layer (output layer)
        x = self.fc2(x)

        return x

def get_model():
    """
    Create and return an instance of the CatDogCNN model.

    Returns:
        CatDogCNN: An instance of the CNN model
    """
    return CatDogCNN()

if __name__ == "__main__":
    # Test the model with a random input
    model = get_model()
    print(model)

    # Create a random input tensor
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # Calculate the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
