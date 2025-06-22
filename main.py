"""
Main Script for Cat vs. Dog Image Classifier

This script demonstrates how to use the project to train and evaluate a model
for cat vs. dog classification.
"""

import os
import argparse
import torch
import numpy as np

from data_loader import get_data_loaders
from model import get_model
from train import train_model
from evaluate import evaluate_model
from utils import load_model, predict_image, visualize_prediction

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cat vs. Dog Image Classifier')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'predict'],
                        help='Mode: train, evaluate, or predict')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and evaluation')
    parser.add_argument('--model_path', type=str, default='./models/best_model.pth',
                        help='Path to save/load model')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to image for prediction')
    return parser.parse_args()

def train(args):
    """Train the model."""
    print("=== Training Mode ===")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get data loaders
    train_loader, val_loader, _ = get_data_loaders(batch_size=args.batch_size)

    # Get model
    model = get_model()
    model = model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )

    # Create directory for saving models if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        save_dir=os.path.dirname(args.model_path)
    )

    print('Training completed successfully!')

def evaluate(args):
    """Evaluate the model."""
    print("=== Evaluation Mode ===")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get data loaders (only need test loader)
    _, _, test_loader = get_data_loaders(batch_size=args.batch_size)

    # Get model
    model = get_model()

    # Load trained model
    try:
        model = load_model(model, args.model_path)
    except FileNotFoundError:
        print(f"Model not found at {args.model_path}. Please train the model first.")
        return

    # Move model to device
    model = model.to(device)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Evaluate model
    test_acc, test_loss, _, _ = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

def predict(args):
    """Make a prediction for a single image."""
    print("=== Prediction Mode ===")

    if args.image_path is None:
        print("Please provide an image path using --image_path")
        return

    if not os.path.exists(args.image_path):
        print(f"Image not found at {args.image_path}")
        return

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get model
    model = get_model()

    # Load trained model
    try:
        model = load_model(model, args.model_path)
    except FileNotFoundError:
        print(f"Model not found at {args.model_path}. Please train the model first.")
        return

    # Move model to device
    model = model.to(device)

    # Make prediction
    predicted_class, confidence = predict_image(model, args.image_path, device=device)

    # Map class index to name
    class_names = ['Cat', 'Dog']
    class_name = class_names[predicted_class]

    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence:.2%}")

    # Visualize prediction
    visualize_prediction(args.image_path, predicted_class, confidence)

def main():
    """Main function."""
    args = parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'predict':
        predict(args)
    else:
        print(f"Invalid mode: {args.mode}")

if __name__ == '__main__':
    main()
