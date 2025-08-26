# =============================================================================
# MNIST Digit Classifier - Professional PyTorch Implementation
# =============================================================================
# 
# This script implements a Multi-Layer Perceptron (MLP) neural network for
# handwritten digit classification using the MNIST dataset. The implementation
# showcases advanced PyTorch concepts including:
#
# - Custom neural network architecture with dropout regularization
# - Professional training pipeline with comprehensive progress tracking
# - Advanced analysis tools for model evaluation
# - Best practices in PyTorch development and code organization
#
# Author: [Your Name]
# GitHub: [Your GitHub]
# License: MIT
# =============================================================================

import torch                              # PyTorch core library for tensors and neural networks
import torch.nn as nn                     # Neural network modules (Linear, Dropout, etc.)
import torch.nn.functional as F           # Functional operations (ReLU, softmax, etc.)
from torch.utils.data import DataLoader   # Efficient batching and iteration over datasets
from torchvision import datasets, transforms  # Standard computer vision datasets and preprocessing
import pickle                             # For saving misclassified examples data

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================
# These hyperparameters control the training process and model behavior
# Adjust these values based on your hardware capabilities and requirements

BATCH_SIZE = 128                          # Number of images processed per training step
                                         # Larger batches = faster training but more memory usage
                                         # Smaller batches = better generalization but slower training

EPOCHS = 7                                # Number of complete passes through the training dataset
                                         # More epochs = potentially better performance but risk of overfitting
                                         # Monitor train/test gap to determine optimal stopping point

LR = 1e-3                                 # Learning rate - step size for the optimizer
                                         # Too high = training may be unstable
                                         # Too low = training may be very slow
                                         # 1e-3 is a good starting point for AdamW

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically use GPU if available
                                                         # GPU training is significantly faster for large models
                                                         # Falls back to CPU if no GPU is detected

torch.manual_seed(0)                      # Set random seed for reproducible results
                                         # Ensures consistent weight initialization and data shuffling
                                         # Important for debugging and comparing different runs

# =============================================================================
# DATA PREPARATION SECTION
# =============================================================================
# The MNIST dataset contains 28x28 grayscale handwritten digit images
# We apply standard preprocessing to normalize the data and improve training

# Define image transformations for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),                # Convert PIL images to PyTorch tensors
                                         # Changes shape from [H, W, C] to [C, H, W]
                                         # Normalizes pixel values from [0, 255] to [0, 1]
    
    transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalization
                                                 # Mean: 0.1307, Standard deviation: 0.3081
                                                 # Helps with training stability and convergence
])

# Load training dataset (60,000 images)
train_ds = datasets.MNIST(
    root="./data",                        # Directory to store/load the dataset
    train=True,                           # Use training split (vs. test split)
    download=True,                        # Download if not already present
    transform=transform                   # Apply preprocessing transformations
)

# Load test dataset (10,000 images)
test_ds = datasets.MNIST(
    root="./data",                        # Directory to store/load the dataset
    train=False,                          # Use test split (vs. training split)
    download=True,                        # Download if not already present
    transform=transform                   # Apply preprocessing transformations
)

# =============================================================================
# MULTI-THREADING CONFIGURATION
# =============================================================================
# Automatically configure multi-threading based on system capabilities
# Users can override with environment variables if needed

import os
import platform

def get_optimal_worker_count():
    """
    Determine optimal number of workers for DataLoader based on system capabilities.
    
    Returns:
        int: Number of workers (0 for single-threaded, >0 for multi-threaded)
    
    Logic:
        1. Check if user explicitly set NUM_WORKERS environment variable
        2. Detect Windows and use 0 (single-threaded) for compatibility
        3. Use CPU count - 1 for other systems (leaving one core free)
        4. Fall back to 0 if detection fails
    """
    # Check if user explicitly set workers
    if 'NUM_WORKERS' in os.environ:
        try:
            workers = int(os.environ['NUM_WORKERS'])
            print(f"Using user-specified workers: {workers}")
            return workers
        except ValueError:
            print("Warning: Invalid NUM_WORKERS value, using auto-detection")
    
    # Auto-detect optimal configuration
    system = platform.system().lower()
    
    if system == 'windows':
        # Windows has known issues with multiprocessing in DataLoader
        # Use single-threaded mode for maximum compatibility
        print("Windows detected: Using single-threaded mode (num_workers=0)")
        print("  → Set NUM_WORKERS environment variable to override if needed")
        return 0
    
    elif system in ['linux', 'darwin']:  # Linux and macOS
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            # Use CPU count - 1, but cap at 8 to avoid overwhelming the system
            optimal_workers = min(cpu_count - 1, 8)
            if optimal_workers > 0:
                print(f"Multi-threading enabled: {optimal_workers} workers (CPU cores: {cpu_count})")
                return optimal_workers
            else:
                print("Single CPU core detected: Using single-threaded mode")
                return 0
        except Exception as e:
            print(f"Could not detect CPU count: {e}")
            print("Falling back to single-threaded mode")
            return 0
    
    else:
        # Unknown system, be conservative
        print(f"Unknown system '{system}': Using single-threaded mode for compatibility")
        return 0

# Get optimal worker count for this system
OPTIMAL_WORKERS = get_optimal_worker_count()

# Create data loaders for efficient batch processing
train_loader = DataLoader(
    train_ds,                            # Dataset to iterate over
    batch_size=BATCH_SIZE,               # Number of samples per batch
    shuffle=True,                        # Randomize order for each epoch (important for SGD)
    num_workers=OPTIMAL_WORKERS,         # Auto-configured based on system
    pin_memory=OPTIMAL_WORKERS > 0       # Enable pin_memory only with multi-threading
)

test_loader = DataLoader(
    test_ds,                             # Dataset to iterate over
    batch_size=BATCH_SIZE,               # Number of samples per batch
    shuffle=False,                       # No shuffling needed for evaluation
    num_workers=OPTIMAL_WORKERS,         # Auto-configured based on system
    pin_memory=OPTIMAL_WORKERS > 0       # Enable pin_memory only with multi-threading
)

# =============================================================================
# NEURAL NETWORK ARCHITECTURE
# =============================================================================
# Multi-Layer Perceptron (MLP) with dropout regularization
# Architecture: Input(784) -> Linear(512) -> ReLU + Dropout(0.3) -> 
#              Linear(256) -> ReLU + Dropout(0.2) -> Linear(10)

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for MNIST digit classification.
    
    This neural network takes 28x28 grayscale images as input and outputs
    probabilities for 10 digit classes (0-9). The architecture includes
    dropout layers to prevent overfitting and improve generalization.
    
    Architecture:
        - Input: 784 features (28*28 flattened image)
        - Hidden Layer 1: 512 neurons with ReLU activation + 30% dropout
        - Hidden Layer 2: 256 neurons with ReLU activation + 20% dropout  
        - Output Layer: 10 neurons (one per digit class)
    """
    
    def __init__(self):
        """
        Initialize the neural network layers and dropout rates.
        
        The dropout rates are strategically chosen:
        - 30% after first layer (more parameters, higher regularization)
        - 20% after second layer (fewer parameters, moderate regularization)
        - No dropout after final layer (preserves output precision)
        """
        super().__init__()                # Initialize parent class (nn.Module)
        
        # Fully connected layers
        self.fc1 = nn.Linear(28*28, 512)  # First layer: 784 input features -> 512 output features
        self.fc2 = nn.Linear(512, 256)    # Second layer: 512 input features -> 256 output features
        self.fc3 = nn.Linear(256, 10)     # Output layer: 256 input features -> 10 output features (digits 0-9)
        
        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(0.3)   # 30% dropout after first hidden layer
        self.dropout2 = nn.Dropout(0.2)   # 20% dropout after second hidden layer
        # Note: No dropout after final layer to preserve output precision

    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28]
               - batch_size: Number of images in the batch
               - 1: Single grayscale channel
               - 28, 28: Image height and width
        
        Returns:
            logits: Raw output scores of shape [batch_size, 10]
                   - 10 scores corresponding to each digit class (0-9)
                   - Higher scores indicate higher confidence for that class
        """
        # x arrives as a batch of images: shape [B, 1, 28, 28]
        x = x.view(x.size(0), -1)         # Flatten to [B, 784] - reshape to 2D tensor
        x = F.relu(self.fc1(x))           # First layer + ReLU activation: [B, 512]
        x = self.dropout1(x)              # Apply 30% dropout for regularization
        x = F.relu(self.fc2(x))           # Second layer + ReLU activation: [B, 256]
        x = self.dropout2(x)              # Apply 20% dropout for regularization
        logits = self.fc3(x)              # Final layer: [B, 10] - raw scores per class
        
        return logits                     # Return logits (CrossEntropyLoss applies softmax internally)

# Create model instance and move to appropriate device (GPU/CPU)
model = MLP().to(DEVICE)                  # Instantiate model and move to GPU/CPU
print(f"Model created and moved to: {DEVICE}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Set up optimizer and loss function for training

# AdamW optimizer with weight decay for regularization
optimizer = torch.optim.AdamW(
    model.parameters(),                   # Parameters to optimize
    lr=LR,                               # Learning rate
    weight_decay=1e-4                    # L2 regularization (weight decay)
                                         # Helps prevent overfitting by penalizing large weights
)

# Cross-entropy loss for multi-class classification
criterion = nn.CrossEntropyLoss()         # Combines LogSoftmax + NLLLoss
                                         # Expects: logits of shape [B, C] and integer labels [B]
                                         # Automatically applies softmax to convert logits to probabilities

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(epoch: int):
    """
    Train the model for one complete epoch.
    
    This function processes all training data once, updating model parameters
    to minimize the loss function. It provides detailed progress tracking
    and returns final metrics for the epoch.
    
    Args:
        epoch: Current epoch number (for logging purposes)
    
    Returns:
        avg_loss: Average training loss across all batches
        acc: Training accuracy across all batches
    """
    model.train()                         # Enable training mode (enables dropout, batch norm updates)
    total_loss, total_correct, total_examples = 0.0, 0, 0
    
    # Calculate progress intervals for real-time monitoring
    total_batches = len(train_loader)     # Total number of batches in this epoch
    progress_interval = max(1, total_batches // 10)  # Update progress every 10% of batches
    
    print(f"\nEpoch {epoch} - Training Progress:")
    print("=" * 50)
    
    # Iterate through all training batches
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to appropriate device (GPU/CPU)
        images = images.to(DEVICE, non_blocking=True)  # Move image batch to device
        labels = labels.to(DEVICE, non_blocking=True)  # Move label batch to device
        
        # Forward pass: compute predictions
        optimizer.zero_grad()             # Clear gradients from previous batch
        logits = model(images)            # Forward pass through network: [B, 10]
        loss = criterion(logits, labels)  # Compute loss between predictions and true labels
        
        # Backward pass: compute gradients
        loss.backward()                   # Compute gradients with respect to all parameters
        optimizer.step()                  # Update parameters using computed gradients
        
        # Accumulate statistics for this batch
        total_loss += loss.item() * images.size(0)      # Sum batch loss * batch size (for averaging)
        preds = logits.argmax(dim=1)                     # Get predicted class (highest logit)
        total_correct += (preds == labels).sum().item()  # Count correct predictions
        total_examples += images.size(0)                 # Track total samples processed
        
        # Progress update every 10% of batches (or at the end)
        if (batch_idx + 1) % progress_interval == 0 or batch_idx == total_batches - 1:
            progress_percent = ((batch_idx + 1) / total_batches) * 100  # Calculate percentage complete
            current_loss = total_loss / total_examples                   # Running average loss so far
            current_acc = total_correct / total_examples                 # Running average accuracy so far
            current_batch_loss = loss.item()                            # Loss for just this batch
            
            # Display comprehensive progress information
            print(f"  {progress_percent:5.1f}% | "
                  f"Batch Loss: {current_batch_loss:.4f} | "
                  f"Running Loss: {current_loss:.4f} | "
                  f"Running Acc: {current_acc:.4f} | "
                  f"Batch {batch_idx + 1}/{total_batches}")

    # Calculate final epoch statistics
    avg_loss = total_loss / total_examples               # Mean loss over all samples in epoch
    acc = total_correct / total_examples                 # Final training accuracy for this epoch
    
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Final Train Loss: {avg_loss:.4f}")
    print(f"  Final Train Acc:  {acc:.4f}")
    
    return avg_loss, acc  # Return values for tracking and summary table

def evaluate():
    """
    Evaluate the model on the test dataset.
    
    This function assesses the model's generalization performance on unseen data.
    It runs in evaluation mode (no gradients, no dropout) and provides
    detailed progress tracking during evaluation.
    
    Returns:
        acc: Test accuracy across all test samples
    """
    model.eval()                         # Enable evaluation mode (disables dropout, fixes batch norm stats)
    total_correct, total_examples = 0, 0
    
    print("\nEvaluating on Test Set:")
    print("-" * 30)
    
    # Calculate progress intervals for evaluation (same 10% logic as training)
    total_test_batches = len(test_loader)               # Total number of test batches
    progress_interval = max(1, total_test_batches // 10)  # Progress update frequency
    
    # Iterate through all test batches
    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Move data to appropriate device
            images = images.to(DEVICE, non_blocking=True)   # Move image batch to device
            labels = labels.to(DEVICE, non_blocking=True)   # Move label batch to device
            
            # Forward pass only (no gradients needed)
            logits = model(images)                          # Get predictions: [B, 10]
            preds = logits.argmax(dim=1)                    # Get predicted class (highest logit)
            
            # Accumulate correct predictions
            total_correct += (preds == labels).sum().item() # Count correct predictions
            total_examples += images.size(0)                # Track total samples processed
            
            # Progress update every 10% of test batches (or at the end)
            if (batch_idx + 1) % progress_interval == 0 or batch_idx == total_test_batches - 1:
                progress_percent = ((batch_idx + 1) / total_test_batches) * 100  # Calculate test progress
                current_acc = total_correct / total_examples                      # Running test accuracy so far
                print(f"  {progress_percent:5.1f}% | "
                      f"Running Test Acc: {current_acc:.4f} | "
                      f"Batch {batch_idx + 1}/{total_test_batches}")

    # Calculate final test accuracy
    acc = total_correct / total_examples                # Test accuracy (generalization metric)
    print(f"\nFinal Test Accuracy: {acc:.4f}")
    
    return acc  # Return just the accuracy

# =============================================================================
# MAIN TRAINING EXECUTION
# =============================================================================
# This section runs the complete training pipeline when the script is executed

if __name__ == '__main__':
    print("=" * 80)
    print("MNIST DIGIT CLASSIFIER - TRAINING STARTED")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LR}")
    print(f"Epochs: {EPOCHS}")
    print(f"Training Samples: {len(train_ds):,}")
    print(f"Test Samples: {len(test_ds):,}")
    print("=" * 80)
    
    # Initialize variables to store final epoch results
    final_train_loss = 0.0                              # Will store the last epoch's training loss
    final_test_accuracy = 0.0                           # Will store the last epoch's test accuracy
    
    # Lists to store epoch statistics for comprehensive summary table
    epoch_stats = []                                    # Will hold data for each epoch
    
    # Main training loop over all epochs
    print(f"\nStarting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):      # Iterate epochs: 1..EPOCHS inclusive
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{EPOCHS}")
        print(f"{'='*60}")
        
        # Train for one complete pass over training data
        train_loss, train_acc = train_one_epoch(epoch)
        
        # Evaluate on held-out test set
        test_acc = evaluate()
        
        # Store comprehensive statistics for this epoch
        epoch_stats.append({
            'epoch': epoch,                             # Current epoch number
            'train_loss': train_loss,                   # Training loss for this epoch
            'train_acc': train_acc,                     # Training accuracy for this epoch
            'test_acc': test_acc,                       # Test accuracy for this epoch
            'gap': train_acc - test_acc                 # Training vs Test accuracy difference (overfitting indicator)
        })
        
        # Store the final epoch values for saving with the model
        if epoch == EPOCHS:                             # Only on the very last epoch
            final_train_loss = train_loss               # Save final training loss
            final_test_accuracy = test_acc              # Save final test accuracy
    
    # =============================================================================
    # COMPREHENSIVE TRAINING SUMMARY
    # =============================================================================
    # Display detailed analysis of training progress and performance
    
    print("\n" + "="*100)
    print("TRAINING SUMMARY TABLE")
    print("="*100)
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Trend':<6} {'Train Acc':<12} {'Trend':<6} {'Test Acc':<12} {'Trend':<6} {'Gap':<8} {'Trend':<6}")
    print("-"*100)
    
    # Process each epoch to show trends and patterns
    for i, stats in enumerate(epoch_stats):
        # Determine trend arrows based on metric changes between epochs
        if i == 0:
            # First epoch - no trend data available
            loss_trend = "-"    # No trend data
            train_trend = "-"   # No trend data
            test_trend = "-"    # No trend data
            gap_trend = "-"     # No trend data
        else:
            # Compare with previous epoch to determine trends
            prev_stats = epoch_stats[i-1]  # Get previous epoch's statistics
            
            # Train Loss trend (lower is better)
            if stats['train_loss'] < prev_stats['train_loss']:
                loss_trend = "DOWN"  # Loss decreasing (good - model is learning)
            elif stats['train_loss'] > prev_stats['train_loss']:
                loss_trend = "UP"    # Loss increasing (bad - model is struggling)
            else:
                loss_trend = "SAME"  # Loss unchanged
            
            # Train Accuracy trend (higher is better)
            if stats['train_acc'] > prev_stats['train_acc']:
                train_trend = "UP"   # Accuracy increasing (good - model is learning)
            elif stats['train_acc'] < prev_stats['train_acc']:
                train_trend = "DOWN" # Accuracy decreasing (bad - model is struggling)
            else:
                train_trend = "SAME" # Accuracy unchanged
            
            # Test Accuracy trend (higher is better)
            if stats['test_acc'] > prev_stats['test_acc']:
                test_trend = "UP"    # Accuracy increasing (good - generalization improving)
            elif stats['test_acc'] < prev_stats['test_acc']:
                test_trend = "DOWN"  # Accuracy decreasing (bad - generalization worsening)
            else:
                test_trend = "SAME"  # Accuracy unchanged
            
            # Gap trend (lower is better for generalization)
            if stats['gap'] < prev_stats['gap']:
                gap_trend = "DOWN"   # Gap decreasing (good - training and test converging)
            elif stats['gap'] > prev_stats['gap']:
                gap_trend = "UP"     # Gap increasing (bad - training and test diverging)
            else:
                gap_trend = "SAME"   # Gap unchanged
        
        # Print comprehensive row for this epoch with all metrics and trends
        print(f"{stats['epoch']:<6} "
              f"{stats['train_loss']:<12.4f} "
              f"{loss_trend:<6} "
              f"{stats['train_acc']:<12.4f} "
              f"{train_trend:<6} "
              f"{stats['test_acc']:<12.4f} "
              f"{test_trend:<6} "
              f"{stats['gap']:<8.4f} "
              f"{gap_trend:<6}")
    
    print("-"*100)
    
    # =============================================================================
    # OVERALL STATISTICS AND ANALYSIS
    # =============================================================================
    # Calculate and display comprehensive performance metrics
    
    # Find best performing epoch
    best_test_acc = max(stats['test_acc'] for stats in epoch_stats)  # Find highest test accuracy
    best_epoch = next(stats['epoch'] for stats in epoch_stats if stats['test_acc'] == best_test_acc)  # Find which epoch achieved it
    
    # Calculate average generalization gap
    avg_gap = sum(stats['gap'] for stats in epoch_stats) / len(epoch_stats)  # Average gap across all epochs
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Best Test Accuracy: {best_test_acc:.4f} (Epoch {best_epoch})")
    print(f"  Average Train-Test Gap: {avg_gap:.4f}")
    print(f"  Final Train-Test Gap: {epoch_stats[-1]['gap']:.4f}")
    
    # Performance interpretation
    if avg_gap < 0.01:
        gap_status = "EXCELLENT - No overfitting detected"
    elif avg_gap < 0.03:
        gap_status = "GOOD - Minimal overfitting"
    elif avg_gap < 0.05:
        gap_status = "FAIR - Some overfitting, consider regularization"
    else:
        gap_status = "POOR - Significant overfitting, increase regularization"
    
    print(f"  Generalization Status: {gap_status}")
    
    # =============================================================================
    # MODEL PERSISTENCE
    # =============================================================================
    # Save the trained model with comprehensive metadata for future use
    
    print("\nSaving model and training statistics...")
    
    # Save the trained model with all tracked values and statistics
    torch.save({
        'model_state_dict': model.state_dict(),          # Save the learned weights and biases
        'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state (for resuming training)
        'epoch': EPOCHS,                                # Save final epoch number
        'train_loss': final_train_loss,                 # Save final training loss
        'test_accuracy': final_test_accuracy,           # Save final test accuracy
        'epoch_stats': epoch_stats,                     # Save all epoch statistics for analysis
        'best_test_accuracy': best_test_acc,            # Save best test accuracy achieved
        'best_epoch': best_epoch,                       # Save which epoch achieved best performance
        'model_config': {                               # Save model configuration
            'architecture': 'MLP',
            'input_size': 784,
            'hidden_sizes': [512, 256],
            'output_size': 10,
            'dropout_rates': [0.3, 0.2],
            'activation': 'ReLU'
        },
        'training_config': {                            # Save training configuration
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'optimizer': 'AdamW',
            'loss_function': 'CrossEntropyLoss',
            'device': str(DEVICE)
        }
    }, 'mnist_classifier.pth')
    
    print("Model saved as 'mnist_classifier.pth'")
    print("Training statistics saved with the model")
    
    # =============================================================================
    # TRAINING COMPLETION SUMMARY
    # =============================================================================
    # Final summary and next steps
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
    print(f"Best Test Accuracy: {best_test_acc:.4f} (Epoch {best_epoch})")
    print(f"Generalization Gap: {epoch_stats[-1]['gap']:.4f}")
    print(f"Model saved for future use")
    print("\nNext steps:")
    print("1. Run 'python view_misidentified.py' to analyze misclassifications")
    print("2. Use the saved model for inference on new images")
    print("3. Experiment with different hyperparameters or architectures")
    print("="*80)