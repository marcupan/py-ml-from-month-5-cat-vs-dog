"""
Training Script for Cat vs. Dog Image Classifier

This module handles the training process for the CNN model.
It includes functions for training, validation, and saving the model.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data_loader import get_data_loaders
from model import get_model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=25, device='cuda', save_dir='./models'):
    """
    Train the model and validate it.

    Args:
        model (nn.Module): The neural network model
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimization algorithm
        scheduler: Learning rate scheduler
        num_epochs (int): Number of epochs to train
        device (str): Device to train on ('cuda' or 'cpu')
        save_dir (str): Directory to save model checkpoints

    Returns:
        model (nn.Module): The trained model
        history (dict): Training history
    """
    # Create directory for saving models if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize history dictionary to store metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # Initialize best validation accuracy for model saving
    best_val_acc = 0.0

    # Start training timer
    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Wrap train_loader with tqdm for progress bar
        train_loader_tqdm = tqdm(train_loader, desc=f'Training')

        # Iterate over training data
        for inputs, labels in train_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Update progress bar
            train_loader_tqdm.set_postfix(loss=loss.item())

        # Calculate epoch statistics
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.float() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')

        # Validation phase
        model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        running_corrects = 0

        # Wrap val_loader with tqdm for progress bar
        val_loader_tqdm = tqdm(val_loader, desc=f'Validation')

        # Iterate over validation data
        for inputs, labels in val_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass (no gradient calculation needed for validation)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Update progress bar
            val_loader_tqdm.set_postfix(loss=loss.item())

        # Calculate epoch statistics
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects.float() / len(val_loader.dataset)

        print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')

        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)

        # Save model if validation accuracy improved
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'Model saved with validation accuracy: {epoch_val_acc:.4f}')

        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_acc'].append(epoch_val_acc.item())

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # Calculate total training time
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_val_acc:.4f}')

    # Plot training history
    plot_training_history(history, save_dir)

    # Load best model weights
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, history

def plot_training_history(history, save_dir):
    """
    Plot training and validation loss and accuracy.

    Args:
        history (dict): Training history
        save_dir (str): Directory to save plots
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def main():
    """Main function to train the model."""
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

    # Get data loaders
    train_loader, val_loader, _ = get_data_loaders(batch_size=32)

    # Get model
    model = get_model()
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=25,
        device=device
    )

    print('Training completed successfully!')

if __name__ == '__main__':
    main()
