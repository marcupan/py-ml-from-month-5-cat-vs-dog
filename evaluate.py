"""
Evaluation Script for Cat vs. Dog Image Classifier

This module handles the evaluation of the trained model on the test set.
It includes functions for computing metrics and visualizing results.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from data_loader import get_data_loaders
from model import get_model
from utils import load_model, visualize_model_predictions


def evaluate_model(model, test_loader, criterion, device='cpu'):
    """
    Evaluate the model on the test set.

    Args:
        model (nn.Module): The trained model
        test_loader (DataLoader): DataLoader for a test data
        criterion (nn.Module): Loss function
        device (str): Device to run evaluation on ('cuda' or 'cpu')

    Returns:
        float: Test accuracy
        float: Test loss
        np.ndarray: True labels
        np.ndarray: Predicted labels
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables
    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_preds = []

    # Evaluate model
    print("Evaluating model on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Store labels and predictions for metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate metrics
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.float() / len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    return test_acc.item(), test_loss, np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(true_labels, pred_labels, class_names=['Cat', 'Dog'], save_dir='./models'):
    """
    Plot confusion matrix.

    Args:
        true_labels (np.ndarray): True labels
        pred_labels (np.ndarray): Predicted labels
        class_names (list): List of class names
        save_dir (str): Directory to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    print("Confusion matrix saved to", os.path.join(save_dir, 'confusion_matrix.png'))


def print_classification_report(true_labels, pred_labels, class_names=['Cat', 'Dog']):
    """
    Print a classification report.

    Args:
        true_labels (np.ndarray): True labels
        pred_labels (np.ndarray): Predicted labels
        class_names (list): List of class names
    """
    # Print classification report
    report = classification_report(true_labels, pred_labels, target_names=class_names)
    print("\nClassification Report:")
    print(report)


def main():
    """Main function to evaluate the model."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device - prioritize GPU (CUDA or MPS) over CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Get data loaders (only need test loader)
    _, _, test_loader = get_data_loaders(batch_size=32)

    # Get model
    model = get_model()

    # Load trained model
    model_path = './models/best_model.pth'
    try:
        model = load_model(model, model_path, device=device)
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    # Model is already on the correct device from load_model

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Evaluate model
    test_acc, test_loss, true_labels, pred_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    # Plot confusion matrix
    plot_confusion_matrix(true_labels, pred_labels)

    # Print classification report
    print_classification_report(true_labels, pred_labels)

    # Visualize model predictions
    print("\nVisualizing model predictions...")
    visualize_model_predictions(model, test_loader, device=device)

    print("\nEvaluation completed successfully!")


if __name__ == '__main__':
    main()
