import torch                              # Import PyTorch core library for tensors and neural networks
from model import MLP                     # Import the MLP class from your training file (model.py)
from torchvision import transforms        # Import image transformation utilities
from PIL import Image                     # Import Python Imaging Library for image handling
import numpy as np                        # Import NumPy for numerical operations

# Function to load the previously saved trained model
def load_model():
    model = MLP()                        # Create a new instance of the MLP model with random weights
    checkpoint = torch.load('mnist_classifier.pth', map_location='cpu')  # Load saved model file to CPU memory
    model.load_state_dict(checkpoint['model_state_dict'])                # Replace random weights with trained weights
    model.eval()                         # Set model to evaluation mode (disables dropout, fixes batch norm stats)
    return model, checkpoint             # Return both the loaded model and metadata

# Function to make predictions on new images
def predict_digit(model, image_tensor):
    with torch.no_grad():                # Disable gradient computation for inference (saves memory and speed)
        output = model(image_tensor)     # Pass the image through the model to get logits
        prediction = output.argmax(dim=1) # Find the index of highest score (predicted digit 0-9)
        confidence = torch.softmax(output, dim=1).max(dim=1)[0]  # Convert logits to probabilities and get confidence
        return prediction.item(), confidence.item()               # Return prediction and confidence as Python numbers

# Main execution block - only runs if this file is executed directly
if __name__ == '__main__':
    # Load the model
    model, checkpoint = load_model()     # Call load_model() to get the trained model and metadata
    print(f"Model loaded! Test accuracy was: {checkpoint['test_accuracy']:.4f}")  # Display the model's performance
    
    # Now you can use the model for predictions
    print("Model is ready for predictions!")  # Confirm the model is ready to use
