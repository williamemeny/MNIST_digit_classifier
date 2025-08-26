# =============================================================================
# MNIST Model Loading and Inference Tool - Professional PyTorch Implementation
# =============================================================================
# 
# This script provides functionality to load a pre-trained MNIST digit classifier
# and use it for making predictions on new images. It demonstrates advanced
# PyTorch concepts including:
#
# - Model loading and state restoration
# - Inference pipeline implementation
# - Confidence scoring and prediction analysis
# - Professional error handling and validation
# - Ready-to-use prediction functions
#
# Author: [Your Name]
# GitHub: [Your GitHub]
# License: MIT
# =============================================================================

import torch                              # PyTorch core library for tensors and neural networks
from model import MLP                     # Import the MLP class from your training file (model.py)
from torchvision import transforms        # Import image transformation utilities for preprocessing
from PIL import Image                     # Import Python Imaging Library for image handling
import numpy as np                        # Import NumPy for numerical operations and array manipulation

# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================

def load_model(checkpoint_path='mnist_classifier.pth', device='auto'):
    """
    Load a pre-trained MNIST classifier model from checkpoint file.
    
    This function loads the saved model weights and configuration from a training
    session. It handles device placement automatically and provides comprehensive
    validation of the loaded model.
    
    Args:
        checkpoint_path (str): Path to the saved model checkpoint file
                              Default: 'mnist_classifier.pth'
        device (str): Device to load the model on ('cpu', 'cuda', or 'auto')
                     Default: 'auto' (automatically selects best available device)
    
    Returns:
        tuple: (model, checkpoint_data)
            - model: Loaded MLP model in evaluation mode
            - checkpoint_data: Dictionary containing training metadata
            
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If model architecture doesn't match checkpoint
        ValueError: If checkpoint file is corrupted or invalid
        
    Example:
        >>> model, checkpoint = load_model()
        >>> print(f"Model loaded with {checkpoint['test_accuracy']:.2%} test accuracy")
    """
    
    # =============================================================================
    # DEVICE SELECTION AND VALIDATION
    # =============================================================================
    
    if device == 'auto':
        # Automatically select the best available device
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"CUDA GPU detected: {torch.cuda.get_device_name()}")
        else:
            device = 'cpu'
            print(f"Using CPU for inference")
    else:
        # Validate user-specified device
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        elif device not in ['cpu', 'cuda']:
            print(f"Invalid device '{device}', using CPU")
            device = 'cpu'
    
    print(f"📱 Loading model on device: {device}")
    
    # =============================================================================
    # MODEL INSTANCE CREATION
    # =============================================================================
    
    try:
        # Create a new instance of the MLP model with random weights
        # This matches the architecture used during training
        model = MLP()
        print(f"Model architecture created successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,} total")
        
    except Exception as e:
        raise RuntimeError(f"Failed to create model instance: {e}")
    
    # =============================================================================
    # CHECKPOINT LOADING AND VALIDATION
    # =============================================================================
    
    try:
        # Load the saved model checkpoint from file
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint with device mapping for compatibility
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Validate checkpoint structure
        required_keys = ['model_state_dict', 'test_accuracy', 'epoch']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")
        
        print(f"Checkpoint loaded successfully")
        print(f"Training completed at epoch: {checkpoint['epoch']}")
        print(f"Test accuracy: {checkpoint['test_accuracy']:.4f}")
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Checkpoint file '{checkpoint_path}' not found!\n"
            "Please ensure you have trained the model first using 'python model.py'"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    # =============================================================================
    # MODEL WEIGHT RESTORATION
    # =============================================================================
    
    try:
        # Load the trained weights into the model
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model weights restored successfully")
        
    except RuntimeError as e:
        # Handle architecture mismatch errors
        if "size mismatch" in str(e):
            raise RuntimeError(
                "Model architecture mismatch! The saved checkpoint has a different "
                "architecture than the current MLP class. This usually happens when:\n"
                "1. The model architecture was changed after training\n"
                "2. Different versions of the code are being used\n"
                "3. The checkpoint is from a different project"
            )
        else:
            raise RuntimeError(f"Failed to load model weights: {e}")
    
    # =============================================================================
    # MODEL PREPARATION FOR INFERENCE
    # =============================================================================
    
    # Move model to the specified device
    model = model.to(device)
    print(f"📱 Model moved to device: {device}")
    
    # Set model to evaluation mode
    model.eval()
    print(f"Model set to evaluation mode (dropout disabled)")
    
    # =============================================================================
    # ADDITIONAL METADATA EXTRACTION
    # =============================================================================
    
    # Extract additional useful information from checkpoint
    if 'best_test_accuracy' in checkpoint:
        print(f"Best test accuracy: {checkpoint['best_test_accuracy']:.4f}")
    
    if 'best_epoch' in checkpoint:
        print(f"Best performance at epoch: {checkpoint['best_epoch']}")
    
    if 'train_loss' in checkpoint:
        print(f"Final training loss: {checkpoint['train_loss']:.4f}")
    
    # =============================================================================
    # MODEL VALIDATION
    # =============================================================================
    
    # Perform basic validation that the model is working
    try:
        with torch.no_grad():
            # Create a dummy input tensor for validation
            dummy_input = torch.randn(1, 1, 28, 28).to(device)
            dummy_output = model(dummy_input)
            
            # Validate output shape and properties
            expected_shape = (1, 10)
            if dummy_output.shape != expected_shape:
                raise ValueError(f"Model output shape {dummy_output.shape} != expected {expected_shape}")
            
            # Check that outputs are finite numbers
            if not torch.isfinite(dummy_output).all():
                raise ValueError("Model produced non-finite outputs")
            
        print(f"Model validation successful")
        
    except Exception as e:
        print(f"Model validation warning: {e}")
        print(f"Model may still work, but proceed with caution")
    
    print(f"Model loading completed successfully!")
    return model, checkpoint

# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def predict_digit(model, image_tensor, return_probabilities=False):
    """
    Make a prediction on a single image using the loaded model.
    
    This function performs inference on a single image, returning the predicted
    digit and confidence score. It handles the complete inference pipeline
    including preprocessing, model forward pass, and post-processing.
    
    Args:
        model: Loaded MLP model in evaluation mode
        image_tensor: Input image tensor of shape [1, 1, 28, 28]
                     Should be normalized and on the same device as the model
        return_probabilities (bool): Whether to return full probability distribution
                                   Default: False (returns only top prediction)
    
    Returns:
        If return_probabilities=False:
            tuple: (predicted_digit, confidence_score)
                - predicted_digit (int): Predicted digit (0-9)
                - confidence_score (float): Confidence in the prediction (0.0-1.0)
        
        If return_probabilities=True:
            tuple: (predicted_digit, confidence_score, probability_distribution)
                - probability_distribution (torch.Tensor): Full probability distribution [10]
    
    Note:
        - Input tensor should be preprocessed (normalized, correct shape)
        - Model should be in evaluation mode
        - Function automatically handles device placement
    """
    
    # =============================================================================
    # INPUT VALIDATION
    # =============================================================================
    
    # Validate input tensor shape and properties
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
    
    expected_shape = (1, 1, 28, 28)
    if image_tensor.shape != expected_shape:
        raise ValueError(f"Expected input shape {expected_shape}, got {image_tensor.shape}")
    
    if not torch.isfinite(image_tensor).all():
        raise ValueError("Input tensor contains non-finite values")
    
    # =============================================================================
    # DEVICE PLACEMENT
    # =============================================================================
    
    # Ensure input tensor is on the same device as the model
    model_device = next(model.parameters()).device
    if image_tensor.device != model_device:
        image_tensor = image_tensor.to(model_device)
    
    # =============================================================================
    # MODEL INFERENCE
    # =============================================================================
    
    with torch.no_grad():  # Disable gradient computation for inference (saves memory and speed)
        try:
            # Forward pass through the model
            output = model(image_tensor)     # Get raw logits: [1, 10]
            
            # Convert logits to probabilities using softmax
            probabilities = torch.softmax(output, dim=1)  # [1, 10]
            
            # Get the predicted digit (highest probability)
            prediction = output.argmax(dim=1)  # [1] -> predicted class index
            
            # Get confidence score (probability of predicted class)
            confidence = probabilities.max(dim=1)[0]  # [1] -> confidence score
            
            # Extract scalar values
            predicted_digit = prediction.item()      # Convert to Python int
            confidence_score = confidence.item()    # Convert to Python float
            
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")
    
    # =============================================================================
    # OUTPUT PROCESSING
    # =============================================================================
    
    if return_probabilities:
        # Return full probability distribution along with prediction
        return predicted_digit, confidence_score, probabilities.squeeze()
    else:
        # Return only the top prediction and confidence
        return predicted_digit, confidence_score

def predict_batch(model, image_batch, return_probabilities=False):
    """
    Make predictions on a batch of images using the loaded model.
    
    This function performs batch inference for efficiency when processing
    multiple images. It's more efficient than calling predict_digit multiple times.
    
    Args:
        model: Loaded MLP model in evaluation mode
        image_batch: Batch of images tensor of shape [batch_size, 1, 28, 28]
        return_probabilities (bool): Whether to return full probability distributions
                                   Default: False
    
    Returns:
        If return_probabilities=False:
            tuple: (predicted_digits, confidence_scores)
                - predicted_digits (list): List of predicted digits (0-9)
                - confidence_scores (list): List of confidence scores (0.0-1.0)
        
        If return_probabilities=True:
            tuple: (predicted_digits, confidence_scores, probability_distributions)
                - probability_distributions (torch.Tensor): Full probability distributions [batch_size, 10]
    """
    
    # Validate batch input
    if not isinstance(image_batch, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
    
    if len(image_batch.shape) != 4 or image_batch.shape[1:] != (1, 28, 28):
        raise ValueError(f"Expected batch shape [N, 1, 28, 28], got {image_batch.shape}")
    
    # Ensure device placement
    model_device = next(model.parameters()).device
    if image_batch.device != model_device:
        image_batch = image_batch.to(model_device)
    
    # Batch inference
    with torch.no_grad():
        output = model(image_batch)
        probabilities = torch.softmax(output, dim=1)
        predictions = output.argmax(dim=1)
        confidences = probabilities.max(dim=1)[0]
        
        # Convert to Python lists
        predicted_digits = predictions.tolist()
        confidence_scores = confidences.tolist()
    
    if return_probabilities:
        return predicted_digits, confidence_scores, probabilities
    else:
        return predicted_digits, confidence_scores

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_info(model, checkpoint):
    """
    Extract and display comprehensive information about the loaded model.
    
    This function provides a detailed overview of the model's architecture,
    performance metrics, and training history.
    
    Args:
        model: Loaded MLP model
        checkpoint: Checkpoint data dictionary
    
    Returns:
        dict: Comprehensive model information
    """
    
    # Get model architecture information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Extract training metrics
    model_info = {
        'architecture': 'MLP',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'test_accuracy': checkpoint.get('test_accuracy', 'N/A'),
        'best_test_accuracy': checkpoint.get('best_test_accuracy', 'N/A'),
        'final_epoch': checkpoint.get('epoch', 'N/A'),
        'best_epoch': checkpoint.get('best_epoch', 'N/A'),
        'final_train_loss': checkpoint.get('train_loss', 'N/A'),
        'device': str(next(model.parameters()).device)
    }
    
    return model_info

def print_model_summary(model, checkpoint):
    """
    Print a formatted summary of the loaded model and its performance.
    
    Args:
        model: Loaded MLP model
        checkpoint: Checkpoint data dictionary
    """
    
    info = get_model_info(model, checkpoint)
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Architecture: {info['architecture']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Test Accuracy: {info['test_accuracy']:.4f}")
    print(f"Best Test Accuracy: {info['best_test_accuracy']:.4f}")
    print(f"Final Epoch: {info['final_epoch']}")
    print(f"Best Epoch: {info['best_epoch']}")
    print(f"Final Train Loss: {info['final_train_loss']:.4f}")
    print(f"Device: {info['device']}")
    print("="*60)

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
# This section runs when the script is executed directly

if __name__ == '__main__':
    print("MNIST MODEL LOADING AND INFERENCE TOOL")
    print("=" * 60)
    
    try:
        # Step 1: Load the pre-trained model
        print("\nStep 1: Loading pre-trained model...")
        model, checkpoint = load_model()
        
        # Step 2: Display comprehensive model information
        print("\nStep 2: Model information...")
        print_model_summary(model, checkpoint)
        
        # Step 3: Demonstrate model readiness
        print("\nStep 3: Model validation...")
        print("Model is ready for predictions!")
        print("You can now use this model for:")
        print("   • Single image predictions (predict_digit function)")
        print("   • Batch predictions (predict_batch function)")
        print("   • Integration into other applications")
        
        # Step 4: Example usage demonstration
        print("\n🔍 Step 4: Example usage...")
        print("Example code:")
        print("  from load_model import load_model, predict_digit")
        print("  model, checkpoint = load_model()")
        print("  digit, confidence = predict_digit(model, your_image_tensor)")
        print("  print(f'Predicted: {digit}, Confidence: {confidence:.3f}')")
        
    except Exception as e:
        print(f"\n Error during model loading: {e}")
        print("\n Troubleshooting tips:")
        print("   1. Ensure 'mnist_classifier.pth' exists in current directory")
        print("   2. Run 'python model.py' first to train the model")
        print("   3. Check that model.py and load_model.py are in same directory")
        print("   4. Verify PyTorch installation: pip install torch torchvision")
        
    print("\n" + "="*60)
    print("Thank you for using the MNIST Model Loading Tool!")
    print("Star this repository if you found it helpful!")
    print("Connect with me for collaboration opportunities!")