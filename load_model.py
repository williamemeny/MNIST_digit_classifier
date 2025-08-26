import torch                              # Import PyTorch core library for tensors and neural networks
import torch.nn as nn                     # Import neural network modules (Linear layers, etc.)
import torch.nn.functional as F           # Import functional operations (ReLU, softmax, etc.)
from torchvision import transforms        # Import image transformation utilities
from PIL import Image                     # Import Python Imaging Library for image handling
import numpy as np                        # Import NumPy for numerical operations

# Define the same model architecture that was used during training
class MLP(nn.Module):                    # Create a class that inherits from PyTorch's Module base class
    def __init__(self):                   # Constructor method - runs when creating a new MLP instance
        super().__init__()                # Call parent class (nn.Module) constructor to set up internals
        self.fc1 = nn.Linear(28*28, 512) # First fully-connected layer: 784 input features -> 512 output features
        self.fc2 = nn.Linear(512, 256)   # Second fully-connected layer: 512 input features -> 256 output features
        self.fc3 = nn.Linear(256, 10)    # Third fully-connected layer: 256 input features -> 10 output features (digits 0-9)
        self.dropout1 = nn.Dropout(0.3)   # Add dropout layer after first layer (30% dropout)
        self.dropout2 = nn.Dropout(0.2)   # Add dropout layer after second layer (20% dropout)

    def forward(self, x):                 # Define how data flows through the network (forward pass)
        x = x.view(x.size(0), -1)        # Reshape input from [batch, 1, 28, 28] to [batch, 784] (flatten)
        x = F.relu(self.fc1(x))          # Pass through first layer, then apply ReLU activation function
        x = self.dropout1(x)              # Apply 30% dropout after first layer
        x = F.relu(self.fc2(x))          # Pass through second layer, then apply ReLU activation function
        x = self.dropout2(x)              # Apply 20% dropout after second layer
        logits = self.fc3(x)             # Pass through final layer to get raw output scores (logits)
        return logits                     # Return the logits (will be converted to probabilities later)

# Function to load the previously saved trained model
def load_model():
    model = MLP()                        # Create a new instance of the MLP model with random weights
    checkpoint = torch.load('mnist_classifier.pth', map_location='cpu')  # Load saved model file to CPU memory
    model.load_state_dict(checkpoint['model_state_dict'])                # Replace random weights with trained weights
    model.eval()                         # Set model to evaluation mode (disables dropout, fixes batch norm stats)
    return model, checkpoint             # Return both the loaded model and the checkpoint data

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
    print("Model is ready for predictions!")  # Confirm the model is loaded and ready to use
