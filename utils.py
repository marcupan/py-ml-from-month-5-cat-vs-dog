"""
Utility Functions for Cat vs. Dog Image Classifier

This module contains helper functions for the project, such as:
- Visualization utilities
- Model loading and saving
- Prediction utilities
- Miscellaneous helper functions
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

def load_model(model, model_path, device=None):
    """
    Load a saved model checkpoint.

    Args:
        model (nn.Module): The model architecture
        model_path (str): Path to the saved model checkpoint
        device (torch.device, optional): Device to load the model to. If None, uses current device.

    Returns:
        model (nn.Module): The model with loaded weights
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Determine the device to use
    if device is None:
        # Use the same device selection logic as in train.py and evaluate.py
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    # Load the checkpoint to the appropriate device
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Explicitly move the model to the device
    model = model.to(device)

    print(f"Model loaded from {model_path}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"Epoch: {checkpoint['epoch']}")

    return model

def predict_image(model, image_path, device='cpu'):
    """
    Make a prediction for a single image.

    Args:
        model (nn.Module): The trained model
        image_path (str): Path to the image file
        device (str): Device to run inference on ('cuda' or 'cpu')

    Returns:
        int: Predicted class (0 for cat, 1 for dog)
        float: Probability of the prediction
    """
    # Set model to evaluation mode
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')

    # Define the same transformation as used for validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transformation and add batch dimension
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence

def visualize_prediction(image_path, predicted_class, confidence, class_names=['Cat', 'Dog']):
    """
    Visualize an image with its prediction.

    Args:
        image_path (str): Path to the image file
        predicted_class (int): Predicted class index
        confidence (float): Prediction confidence
        class_names (list): List of class names
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Create figure
    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    # Add prediction information
    class_name = class_names[predicted_class]
    title = f"Prediction: {class_name} ({confidence:.2%})"
    plt.title(title, fontsize=16)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_model_predictions(model, data_loader, class_names=['Cat', 'Dog'],
                               num_images=6, device='cpu'):
    """
    Visualize model predictions on a batch of images.

    Args:
        model (nn.Module): The trained model
        data_loader (DataLoader): DataLoader containing images
        class_names (list): List of class names
        num_images (int): Number of images to visualize
        device (str): Device to run inference on ('cuda' or 'cpu')
    """
    # Set model to evaluation mode
    model.eval()

    # Get a batch of images
    images, labels = next(iter(data_loader))

    # Make predictions
    with torch.no_grad():
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # Convert images for display
    images = images.cpu().numpy().transpose((0, 2, 3, 1))

    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)

    # Plot images
    fig, axes = plt.subplots(2, num_images//2, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
        axes[i].axis('off')

        # Add green/red border based on correct/incorrect prediction
        if preds[i] == labels[i]:
            # Green border for correct prediction
            for spine in axes[i].spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        else:
            # Red border for incorrect prediction
            for spine in axes[i].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)

    plt.tight_layout()
    plt.show()

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model (nn.Module): The model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_lr(optimizer):
    """
    Get the current learning rate from the optimizer.

    Args:
        optimizer: The optimizer

    Returns:
        float: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == "__main__":
    # Test the utility functions
    from model import get_model

    # Create a model
    model = get_model()

    # Count parameters
    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params:,}")
