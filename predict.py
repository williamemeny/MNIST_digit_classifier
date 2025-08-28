# =============================================================================
# MNIST Test Image Prediction Tool - Interactive Web Interface
# =============================================================================
# 
# This Flask web application provides an interactive interface for testing
# the trained MNIST digit classifier with random test images from the dataset.
#
# Key Features:
# - 5x5 grid of random MNIST test images (left side - 50% width)
# - Click-to-select functionality for choosing test images
# - Real-time model loading with status confirmation
# - Model parameter display and performance metrics
# - Prediction results with confidence scores
# - Reset button to load new random test images
# - Professional, responsive web interface
#
# Author: William Emeny
# GitHub: https://github.com/williamemeny
# License: MIT
# =============================================================================

import os
import sys
import random
import base64
import io
from typing import Tuple, Dict, Any, Optional

# Flask web framework for the web interface
from flask import Flask, render_template, request, jsonify, send_from_directory

# PyTorch and image processing
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import our existing model loading functionality
from load_model import load_model, predict_digit, get_model_info

# =============================================================================
# FLASK APPLICATION CONFIGURATION
# =============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mnist_test_predictor_2024'

# Global variables for model and data management
model = None
checkpoint = None
model_loaded = False
test_images = []
test_labels = []
current_selection = None
current_displayed_images = []  # Track currently displayed images for selection

# =============================================================================
# MODEL MANAGEMENT FUNCTIONS
# =============================================================================

def initialize_model() -> Tuple[bool, str]:
    """
    Initialize the MNIST model for prediction.
    
    Returns:
        Tuple[bool, str]: (success_status, message)
    """
    global model, checkpoint, model_loaded
    
    try:
        print("🔄 Initializing MNIST model...")
        model, checkpoint = load_model()
        model_loaded = True
        print("✅ Model initialized successfully")
        return True, "Model loaded successfully"
        
    except FileNotFoundError as e:
        error_msg = (
            "Model checkpoint not found! Please train the model first by running:\n"
            "python model.py"
        )
        print(f"❌ {error_msg}")
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg

def get_model_status() -> Dict[str, Any]:
    """
    Get comprehensive model status and information.
    
    Returns:
        Dict containing model status, parameters, and performance metrics
    """
    if not model_loaded:
        return {
            'loaded': False,
            'message': 'Model not loaded'
        }
    
    try:
        # Get detailed model information
        info = get_model_info(model, checkpoint)
        
        return {
            'loaded': True,
            'message': 'Model ready for predictions',
            'info': info
        }
    except Exception as e:
        return {
            'loaded': True,
            'message': f'Model loaded but info unavailable: {str(e)}',
            'info': {}
        }

# =============================================================================
# MNIST DATA LOADING AND PREPROCESSING
# =============================================================================

def load_mnist_test_data() -> Tuple[bool, str]:
    """
    Load MNIST test dataset for random image selection.
    
    Returns:
        Tuple[bool, str]: (success_status, message)
    """
    global test_images, test_labels
    
    try:
        print("📊 Loading MNIST test data...")
        
        # Import here to avoid circular imports
        from torchvision import datasets, transforms
        
        # Define the same transformations used during training
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load test dataset
        test_dataset = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )
        
        # Extract all test images and labels
        test_images = []
        test_labels = []
        
        for i in range(len(test_dataset)):
            img_tensor, label = test_dataset[i]
            test_images.append(img_tensor)
            # Handle both tensor and int label types
            if hasattr(label, 'item'):
                test_labels.append(label.item())
            else:
                test_labels.append(label)
        
        print(f"✅ Loaded {len(test_images)} test images")
        return True, f"Loaded {len(test_images)} test images"
        
    except Exception as e:
        error_msg = f"Failed to load test data: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg

def get_random_test_images(count: int = 25) -> Tuple[list, list]:
    """
    Get random test images for display in the grid.
    
    Args:
        count (int): Number of random images to select
        
    Returns:
        Tuple[list, list]: (image_tensors, labels)
    """
    if len(test_images) == 0:
        return [], []
    
    # Select random indices
    indices = random.sample(range(len(test_images)), min(count, len(test_images)))
    
    # Get corresponding images and labels
    selected_images = [test_images[i] for i in indices]
    selected_labels = [test_labels[i] for i in indices]
    
    return selected_images, selected_labels

def tensor_to_base64_image(tensor: torch.Tensor, size: int = 100) -> str:
    """
    Convert PyTorch tensor to base64-encoded image for web display.
    
    Args:
        tensor (torch.Tensor): Image tensor of shape [1, 28, 28]
        size (int): Output image size in pixels
        
    Returns:
        str: Base64-encoded image data
    """
    try:
        # Convert tensor to numpy array and denormalize
        img_array = tensor.squeeze().numpy()
        
        # Denormalize from MNIST normalization
        img_array = (img_array * 0.3081) + 0.1307
        
        # Clip to valid range [0, 1]
        img_array = np.clip(img_array, 0, 1)
        
        # Convert to PIL Image
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        
        # Resize for web display
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        print(f"Error converting tensor to image: {e}")
        return ""

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_selected_image(image_index: int) -> Dict[str, Any]:
    """
    Predict the digit in the selected test image.
    
    Args:
        image_index (int): Index of the selected image in current grid
        
    Returns:
        Dict containing prediction results and metadata
    """
    global current_selection, test_images, test_labels
    
    if not model_loaded:
        return {
            'success': False,
            'error': 'Model not loaded'
        }
    
    if current_selection is None:
        return {
            'success': False,
            'error': 'No image selected'
        }
    
    try:
        # Get the selected image tensor
        selected_image = current_selection['image_tensor']
        
        # Ensure correct shape for model input
        if selected_image.dim() == 2:
            selected_image = selected_image.unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
        elif selected_image.dim() == 3:
            selected_image = selected_image.unsqueeze(0)  # [1, 1, 28, 28]
        
        # Make prediction
        predicted_digit, confidence = predict_digit(model, selected_image)
        
        # Get actual label
        actual_label = current_selection['label']
        
        # Determine if prediction is correct
        is_correct = predicted_digit == actual_label
        
        return {
            'success': True,
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'actual_digit': actual_label,
            'is_correct': is_correct,
            'confidence_percentage': f"{confidence:.1%}"
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    """Main page route."""
    return render_template('index.html')

@app.route('/api/model/status')
def api_model_status():
    """API endpoint to get model status."""
    status = get_model_status()
    return jsonify(status)

@app.route('/api/images/random')
def api_random_images():
    """API endpoint to get random test images."""
    global test_images, test_labels, current_displayed_images
    
    if len(test_images) == 0:
        return jsonify({'success': False, 'error': 'Test data not loaded'})
    
    try:
        # Get random images
        selected_images, selected_labels = get_random_test_images(25)
        
        # Store the currently displayed images for selection
        current_displayed_images = list(zip(selected_images, selected_labels))
        
        # Convert to base64 for web display
        image_data = []
        for i, (img_tensor, label) in enumerate(zip(selected_images, selected_labels)):
            base64_img = tensor_to_base64_image(img_tensor, size=80)
            image_data.append({
                'index': i,
                'image': base64_img,
                'label': label
            })
        
        return jsonify({
            'success': True,
            'images': image_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/images/select', methods=['POST'])
def api_select_image():
    """API endpoint to select an image for prediction."""
    global current_selection, current_displayed_images
    
    try:
        data = request.get_json()
        image_index = data.get('image_index')
        label = data.get('label')
        
        if image_index is None or label is None:
            return jsonify({'success': False, 'error': 'Missing required data'})
        
        # Get the actual tensor from our currently displayed images
        if image_index < 0 or image_index >= len(current_displayed_images):
            return jsonify({'success': False, 'error': 'Invalid image index'})
        
        selected_tensor, selected_label = current_displayed_images[image_index]
        
        # Verify label matches
        if selected_label != label:
            return jsonify({'success': False, 'error': 'Label mismatch'})
        
        # Store selection
        current_selection = {
            'index': image_index,
            'image_tensor': selected_tensor.clone(),
            'label': label
        }
        
        return jsonify({'success': True, 'message': 'Image selected'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint to predict the selected image."""
    result = predict_selected_image(0)  # Index not used in current implementation
    return jsonify(result)

@app.route('/api/reset', methods=['POST'])
def api_reset():
    """API endpoint to reset and get new random images."""
    global current_selection, current_displayed_images
    
    # Clear current selection and displayed images
    current_selection = None
    current_displayed_images = []
    
    # Return new random images
    return api_random_images()

# =============================================================================
# STATIC FILE SERVING
# =============================================================================

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    print("MNIST Test Image Prediction Tool")
    print("=" * 60)
    
    try:
        # Step 1: Initialize the model
        print("\nStep 1: Initializing model...")
        model_success, model_msg = initialize_model()
        if not model_success:
            print(f"❌ Model initialization failed: {model_msg}")
            print("Please ensure you have trained the model first.")
            sys.exit(1)
        
        # Step 2: Load test data
        print("\nStep 2: Loading test data...")
        data_success, data_msg = load_mnist_test_data()
        if not data_success:
            print(f"❌ Data loading failed: {data_msg}")
            sys.exit(1)
        
        # Step 3: Start Flask application
        print("\nStep 3: Starting web server...")
        print("✅ Application ready!")
        print("🌐 Open your browser and navigate to: http://localhost:5000")
        print("📱 The interface will show:")
        print("   • Left side: 5x5 grid of random MNIST test images")
        print("   • Right side: Model status and prediction results")
        print("   • Click any image to select it, then click 'Predict'")
        print("   • Use 'Reset' to get new random images")
        print("\n📞 Contact: williamgeorgeemeny@gmail.com")
        print("🐦 Twitter: https://x.com/Maths_Master")
        print("💼 GitHub: https://github.com/williamemeny")
        print("\n" + "="*60)

        # Start the Flask development server
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
