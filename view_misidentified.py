# =============================================================================
# MNIST Misclassification Analysis Tool - Professional PyTorch Implementation
# =============================================================================
# 
# This script provides comprehensive analysis and visualization of misclassified
# examples from a trained MNIST digit classifier. It demonstrates advanced
# PyTorch analysis techniques including:
#
# - Loading and analyzing misclassified examples
# - Pattern recognition in model errors
# - Professional visualization using PIL for perfect spacing
# - Comprehensive error analysis and reporting
#
# Author: William Emeny
# GitHub: https://github.com/williamemeny
# License: MIT
# =============================================================================

import pickle                             # For loading saved misclassified examples data
import matplotlib.pyplot as plt           # For optional matplotlib visualization
import numpy as np                        # For numerical operations and array manipulation
from collections import defaultdict       # For grouping misclassified examples by prediction
import webbrowser                         # For automatically opening visualization in browser
import os                                 # For file path operations
from PIL import Image, ImageDraw, ImageFont  # For creating professional visualizations
import torch                              # For tensor operations (if needed)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_misclassified_data():
    """
    Load the saved misclassified examples from the training session.
    
    This function attempts to load the 'misidentified_images.pkl' file that
    was created during the final evaluation of training. The file contains
    all examples where the model made incorrect predictions.
    
    Returns:
        data: Dictionary containing misclassified examples data, or None if file not found
        
    Raises:
        FileNotFoundError: If the pickle file doesn't exist (training hasn't been run)
    """
    try:
        # Attempt to open and load the misclassified examples file
        with open('misidentified_images.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Confirm successful loading with summary
        print(f"Successfully loaded {len(data['images'])} misclassified examples")
        print(f"Data structure: {list(data.keys())}")
        
        return data
        
    except FileNotFoundError:
        # Provide helpful error message if file doesn't exist
        print("Error: 'misidentified_images.pkl' not found!")
        print("�� This usually means:")
        print("   1. Training hasn't been completed yet")
        print("   2. The file was moved or deleted")
        print("   3. Training completed but file wasn't saved")
        print("\nSolution: Run 'python model.py' first to train the model")
        return None
        
    except Exception as e:
        # Handle other potential errors (corrupted file, etc.)
        print(f"Error loading misclassified data: {e}")
        print("The file may be corrupted. Try retraining the model.")
        return None

# =============================================================================
# MISCLASSIFICATION ANALYSIS FUNCTIONS
# =============================================================================

def analyze_misclassifications(data):
    """
    Perform comprehensive analysis of misclassified examples.
    
    This function analyzes the patterns in model errors, grouping examples
    by what the model predicted incorrectly. It provides insights into:
    - Which digits are most commonly misclassified
    - What the model confuses them with
    - Confidence levels of wrong predictions
    - Systematic error patterns
    
    Args:
        data: Dictionary containing misclassified examples data
        
    Returns:
        prediction_groups: Dictionary grouping examples by predicted digit
        
    Note:
        This analysis helps identify model weaknesses and areas for improvement
    """
    if not data:
        print("No data to analyze")
        return None
    
    # Extract data components for analysis
    images = data['images']               # List of misclassified image tensors
    true_labels = data['true_labels']     # List of true digit labels
    predictions = data['predictions']     # List of model's incorrect predictions
    confidences = data['confidences']     # List of confidence scores for wrong predictions
    
    print(f"\n{'='*80}")
    print("MISCLASSIFICATION ANALYSIS")
    print(f"{'='*80}")
    
    # Group misclassified examples by what the model predicted
    # This helps identify systematic errors and confusion patterns
    prediction_groups = defaultdict(list)
    
    for i in range(len(images)):
        pred = predictions[i].item()      # Extract predicted digit (0-9)
        true = true_labels[i].item()     # Extract true digit (0-9)
        confidence = confidences[i].item() # Extract confidence score (0.0-1.0)
        
        # Group by prediction to see what the model confuses
        prediction_groups[pred].append({
            'image': images[i],           # Store the misclassified image
            'true_label': true,           # Store the true label
            'confidence': confidence      # Store the confidence score
        })
    
    # =============================================================================
    # COMPREHENSIVE ERROR ANALYSIS REPORT
    # =============================================================================
    
    print(f"\nMISCLASSIFICATIONS BY PREDICTED DIGIT:")
    print(f"{'Predicted':<10} {'Count':<8} {'True Labels (count)':<30} {'Avg Confidence':<15}")
    print("-" * 80)
    
    # Analyze each prediction group to identify patterns
    for pred_digit in sorted(prediction_groups.keys()):
        examples = prediction_groups[pred_digit]  # Get all examples for this prediction
        count = len(examples)                    # Count of examples in this group
        
        # Count how many times each true label appears for this prediction
        true_label_counts = defaultdict(int)
        for ex in examples:
            true_label_counts[ex['true_label']] += 1
        
        # Format the true label counts for display
        # Shows "digit(count)" for each true label that was misclassified as this prediction
        true_labels_str = ", ".join([f"{label}({count})" for label, count in sorted(true_label_counts.items())])
        
        # Calculate average confidence for this prediction group
        avg_confidence = sum(ex['confidence'] for ex in examples) / count
        
        # Display comprehensive analysis for this prediction
        print(f"{pred_digit:<10} {count:<8} {true_labels_str:<30} {avg_confidence:<15.4f}")
    
    # =============================================================================
    # PATTERN ANALYSIS AND INSIGHTS
    # =============================================================================
    
    print(f"\nPATTERN ANALYSIS:")
    
    # Identify common confusion patterns
    common_confusions = []
    for pred_digit, examples in prediction_groups.items():
        if len(examples) > 1:  # Only analyze predictions with multiple errors
            # Find most common true label for this prediction
            true_label_counts = defaultdict(int)
            for ex in examples:
                true_label_counts[ex['true_label']] += 1
            
            most_common_true = max(true_label_counts.items(), key=lambda x: x[1])
            common_confusions.append((pred_digit, most_common_true[0], most_common_true[1], len(examples)))
    
    # Sort by frequency and display insights
    common_confusions.sort(key=lambda x: x[3], reverse=True)
    
    print(f"Most Common Confusion Patterns:")
    for pred, true, count, total in common_confusions[:5]:  # Show top 5
        percentage = (count / total) * 100
        print(f"  Predicted {pred} → True {true}: {count}/{total} ({percentage:.1f}%)")
    
    # =============================================================================
    # CONFIDENCE ANALYSIS
    # =============================================================================
    
    print(f"\nCONFIDENCE ANALYSIS:")
    
    # Analyze confidence levels of wrong predictions
    all_confidences = [ex['confidence'] for group in prediction_groups.values() for ex in group]
    
    if all_confidences:
        avg_confidence = np.mean(all_confidences)
        high_confidence_errors = sum(1 for c in all_confidences if c > 0.8)
        low_confidence_errors = sum(1 for c in all_confidences if c < 0.5)
        
        print(f"  Average confidence of wrong predictions: {avg_confidence:.3f}")
        print(f"  High confidence errors (>80%): {high_confidence_errors}/{len(all_confidences)}")
        print(f"  Low confidence errors (<50%): {low_confidence_errors}/{len(all_confidences)}")
        
        # Interpretation of confidence patterns
        if avg_confidence > 0.7:
            print(f"Model is overconfident in wrong predictions")
        elif avg_confidence < 0.4:
            print(f"Model shows appropriate uncertainty in wrong predictions")
        else:
            print(f"Model shows moderate confidence in wrong predictions")
    
    return prediction_groups

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def tensor_to_pil(tensor):
    """
    Convert PyTorch tensor to PIL Image for visualization.
    
    This function handles the conversion from PyTorch tensors (which may be
    normalized) back to PIL Images that can be displayed and saved.
    
    Args:
        tensor: PyTorch tensor of shape [1, 28, 28] or [28, 28]
        
    Returns:
        PIL.Image: Grayscale image ready for visualization
        
    Note:
        Handles normalization reversal and proper data type conversion
    """
    # Convert tensor to numpy array
    img_array = tensor.squeeze().numpy()
    
    # Reverse normalization: convert from [0,1] back to [0,255]
    # This assumes the original normalization used in training
    img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
    
    # Convert to PIL Image (grayscale mode 'L')
    return Image.fromarray(img_array, mode='L')

def create_visualization_grid(prediction_groups):
    """
    Create a professional visualization grid using PIL for perfect spacing control.
    
    This function creates a comprehensive visualization that groups misclassified
    examples by what the model predicted. It uses PIL for precise control over
    layout and spacing, ensuring no overlap between text and images.
    
    Args:
        prediction_groups: Dictionary grouping examples by predicted digit
        
    Returns:
        PIL.Image: Composed visualization image
        
    Features:
        - Perfect spacing control (no overlap)
        - Professional layout with clear grouping
        - Comprehensive labeling (True, Predicted, Confidence)
        - High-resolution output suitable for presentations
    """
    if not prediction_groups:
        print("No data to visualize")
        return None
    
    print(f"\nCreating professional visualization...")
    
    # =============================================================================
    # LAYOUT CALCULATIONS
    # =============================================================================
    
    # Calculate dimensions for optimal layout
    max_examples_per_group = max(len(examples) for examples in prediction_groups.values())
    num_groups = len(prediction_groups)
    
    # Fixed dimensions for each element (in pixels)
    IMAGE_SIZE = 100                      # 100x100 pixels for each MNIST image
    TEXT_HEIGHT = 80                      # 80 pixels height for text area below each image
    ROW_SPACING = 40                      # 40 pixels between rows (prediction groups)
    COL_SPACING = 20                      # 20 pixels between columns (examples)
    LEFT_MARGIN = 120                     # 120 pixels left margin (prevents overlap with row labels)
    TOP_MARGIN = 100                      # 100 pixels top margin for title
    
    # Calculate total canvas size needed
    canvas_width = LEFT_MARGIN + max_examples_per_group * (IMAGE_SIZE + COL_SPACING) + 50  # Right margin
    canvas_height = TOP_MARGIN + num_groups * (IMAGE_SIZE + TEXT_HEIGHT + ROW_SPACING) + 50  # Bottom margin
    
    print(f"Canvas dimensions: {canvas_width} x {canvas_height} pixels")
    print(f"Images per row: {max_examples_per_group}")
    print(f"Number of rows: {num_groups}")
    
    # =============================================================================
    # CANVAS CREATION AND SETUP
    # =============================================================================
    
    # Create blank white canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to use professional fonts, fall back to default if not available
    try:
        font_large = ImageFont.truetype("arial.ttf", 24)    # For main title
        font_medium = ImageFont.truetype("arial.ttf", 16)   # For row labels
        font_small = ImageFont.load_default()               # For image labels
        print("sing professional fonts")
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
        print("Using default fonts")
    
    # =============================================================================
    # TITLE AND HEADER
    # =============================================================================
    
    # Add main title
    title = "Misclassified MNIST Examples Grouped by Prediction"
    subtitle = "(True Label | Predicted Label | Confidence)"
    
    # Calculate title position (centered)
    title_bbox = draw.textbbox((0, 0), title, font=font_large)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (canvas_width - title_width) // 2
    draw.text((title_x, 20), title, fill='black', font=font_large)
    
    # Calculate subtitle position (centered)
    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=font_medium)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    subtitle_x = (canvas_width - subtitle_width) // 2
    draw.text((subtitle_x, 50), subtitle, fill='black', font=font_medium)
    
    # =============================================================================
    # IMAGE AND TEXT PLACEMENT
    # =============================================================================
    
    # Position images and text in organized grid
    current_y = TOP_MARGIN
    
    for row_idx, (pred_digit, examples) in enumerate(sorted(prediction_groups.items())):
        print(f"  Processing row {row_idx + 1}/{num_groups}: Predicted {pred_digit}")
        
        # Add row label (prediction group header)
        row_label = f"Predicted {pred_digit}"
        draw.text((20, current_y + IMAGE_SIZE//2 - 10), row_label, fill='black', font=font_medium)
        
        # Position each image in the row
        current_x = LEFT_MARGIN
        
        for col_idx, example in enumerate(examples):
            # Convert PyTorch tensor to PIL image
            pil_image = tensor_to_pil(example['image'])
            
            # Resize to desired dimensions using high-quality resampling
            pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            
            # Paste image onto canvas at calculated position
            canvas.paste(pil_image, (current_x, current_y))
            
            # Add comprehensive text labels below the image
            text_y = current_y + IMAGE_SIZE + 5
            
            # True label (what the digit actually is)
            true_text = f"True: {example['true_label']}"
            draw.text((current_x + 5, text_y), true_text, fill='black', font=font_small)
            
            # Predicted label (what the model incorrectly predicted)
            pred_text = f"Pred: {pred_digit}"
            draw.text((current_x + 5, text_y + 20), pred_text, fill='black', font=font_small)
            
            # Confidence score (how confident the model was in its wrong prediction)
            conf_text = f"Conf: {example['confidence']:.3f}"
            draw.text((current_x + 5, text_y + 40), conf_text, fill='black', font=font_small)
            
            # Move to next column position
            current_x += IMAGE_SIZE + COL_SPACING
        
        # Move to next row position
        current_y += IMAGE_SIZE + TEXT_HEIGHT + ROW_SPACING
    
    # =============================================================================
    # OUTPUT AND DISPLAY
    # =============================================================================
    
    # Save the composed image with high quality
    output_filename = 'misclassified_analysis.png'
    canvas.save(output_filename, 'PNG', dpi=(300, 300))
    print(f"Visualization saved as '{output_filename}'")
    print(f"Resolution: {canvas_width} x {canvas_height} pixels")
    print(f"DPI: 300 (print quality)")
    
    # Automatically open in default browser for immediate viewing
    try:
        webbrowser.open('file://' + os.path.abspath(output_filename))
        print("Opened visualization in browser")
    except:
        print("Could not open browser automatically")
        print(f"Manual: Open '{output_filename}' in your image viewer")
    
    return canvas

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main function to run the complete misclassification analysis.
    
    This function orchestrates the entire analysis pipeline:
    1. Load misclassified examples data
    2. Perform comprehensive pattern analysis
    3. Create professional visualization
    4. Provide actionable insights
    
    The analysis helps identify:
    - Model weaknesses and confusion patterns
    - Areas for improvement in training
    - Data quality issues
    - Potential architecture improvements
    """
    print("MNIST MISCLASSIFICATION ANALYSIS TOOL")
    print("=" * 60)
    
    # Step 1: Load the misclassified examples data
    print("\nStep 1: Loading misclassified examples...")
    data = load_misclassified_data()
    
    if not data:
        print("Cannot proceed without data. Please run training first.")
        return
    
    # Step 2: Perform comprehensive analysis
    print("\nStep 2: Analyzing misclassification patterns...")
    prediction_groups = analyze_misclassifications(data)
    
    if not prediction_groups:
        print("Analysis failed. Please check your data.")
        return
    
    # Step 3: Create professional visualization
    print("\nStep 3: Creating visualization...")
    create_visualization_grid(prediction_groups)
    
    # Step 4: Provide summary and next steps
    print("\nANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("What you now have:")
    print("   • Comprehensive error analysis report")
    print("   • Professional visualization grid")
    print("   • Pattern recognition insights")
    print("   • Confidence analysis")
    print("\nNext steps for improvement:")
    print("   1. Review common confusion patterns")
    print("   2. Consider data augmentation for problematic digits")
    print("   3. Experiment with different architectures")
    print("   4. Adjust regularization parameters")
    print("   5. Collect more training data for weak classes")
    print("\nThe visualization shows exactly where your model struggles!")
    print("=" * 60)

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
# This section runs the analysis when the script is executed directly

if __name__ == "__main__":
    # Run the complete analysis pipeline
    main()
    
    print("\nThank you for using the MNIST Misclassification Analysis Tool!")
    print("Star this repository if you found it helpful!")
    print("\n📞 Contact: williamgeorgeemeny@gmail.com")
    print("🐦 Twitter: https://x.com/Maths_Master")
    print("💼 GitHub: https://github.com/williamemeny")