# 🚀 Professional MNIST Digit Classifier - Portfolio Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-black.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-williamemeny-181717.svg)](https://github.com/williamemeny)

## 📋 Professional Overview

This comprehensive **MNIST Digit Classifier** is a demo project that demonstrates fluency with PyTorch in training neural nets, web development, and data analysis. The project showcases professional software engineering practices while achieving near state-of-the-art performance on the well-known MNIST handwritten digit recognition benchmark.

### 🎯 What This Project Demonstrates

- **🔬 Neural Network Training** - Custom PyTorch implementation with strategic regularization
- **🌐 Full-Stack Web Development** - Modern Flask application with interactive UI
- **📊 Data Analysis & Visualization** - Professional error analysis and pattern recognition
- **🏭 Well-Structured Code** - Model persistence, comprehensive error handling, and deployment
- **⚡ Performance Optimization** - GPU acceleration, batch processing, and scalability

## ✨ Key Features

### 🧠 Neural Network Training
- **Custom MLP Architecture**: 784 → 512 → 256 → 10 neurons with strategic dropout regularization
- **Professional Training Pipeline**: Comprehensive progress tracking, validation, and analysis
- **Advanced Optimization**: AdamW optimizer with weight decay and learning rate scheduling
- **Automatic Device Detection**: Seamless CPU/GPU switching for optimal performance

### 🌐 Interactive Web Interface
- **5×5 Test Image Grid**: Click-to-select functionality for testing random MNIST samples
- **Real-Time Predictions**: Instant digit classification with confidence scores
- **Professional UI/UX**: Modern responsive design with smooth animations
- **RESTful API**: Clean backend architecture with comprehensive error handling

### 📈 Advanced Analysis Tools
- **Misclassification Analysis**: Pattern recognition in model errors and confusion matrices
- **Professional Visualization**: High-quality PIL-based image grids with detailed annotations
- **Performance Metrics**: Comprehensive accuracy tracking and generalization analysis
- **Error Pattern Recognition**: Systematic identification of model weaknesses

### 🛠️ Production Features
- **Model Persistence**: Complete checkpointing with metadata and training statistics
- **Comprehensive Logging**: Detailed progress tracking and debugging information
- **Error Recovery**: Graceful handling of edge cases and system failures
- **Scalable Architecture**: Multi-threading support and batch processing capabilities

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.13+
- Flask 2.3+
- NumPy, Pillow, Matplotlib

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/williamemeny/MNIST_digit_classifier.git
   cd MNIST_digit_classifier
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**
   ```bash
   python model.py
   ```
   *Expected training time: 2-5 minutes on modern hardware*

### Running the Applications

#### 🎯 Web Interface (Recommended)
```bash
python predict.py
```
Navigate to `http://localhost:5000` for the interactive web interface.

#### 🔍 Analysis Tools
```bash
python view_misidentified.py
```
Generates comprehensive misclassification analysis and visualization.

#### 🧠 Model Inference
```bash
python load_model.py
```
Loads and validates the trained model for inference.

## 🏗️ Architecture & Technical Details

### Neural Network Architecture
```
Input Layer: 784 neurons (28×28 flattened image)
├── Hidden Layer 1: 512 neurons + ReLU + 30% Dropout
├── Hidden Layer 2: 256 neurons + ReLU + 20% Dropout
└── Output Layer: 10 neurons (digit classes 0-9)
```

### Training Configuration
- **Batch Size**: 128 (balanced memory/performance)
- **Learning Rate**: 0.001 with AdamW optimization
- **Epochs**: 7 with comprehensive validation
- **Regularization**: Weight decay (1e-4) + dropout
- **Loss Function**: Cross-entropy with softmax

### System Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Training      │    │   Web Interface  │    │   Analysis      │
│   (model.py)    │    │   (predict.py)   │    │   (analysis)    │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Neural Net    │    │ • Flask App      │    │ • Error Analysis│
│ • Data Loading  │    │ • REST API       │    │ • Visualization │
│ • Optimization  │    │ • HTML/CSS/JS    │    │ • Pattern Recog │
│ • Validation    │    │ • Real-time UI   │    │ • Metrics       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Performance Results

### Training Achievements
- **Test Accuracy**: >98% (excellent performance on MNIST benchmark)
- **Training Stability**: Consistent convergence with minimal overfitting
- **Generalization Gap**: <2% (excellent generalization performance)
- **Training Time**: ~3 minutes on modern GPU hardware

### Model Metrics
- **Parameters**: ~535K trainable parameters
- **Model Size**: ~1.6MB (compressed checkpoint)
- **Inference Speed**: <10ms per image
- **Memory Usage**: <500MB during training

### Error Analysis Insights
- Identifies systematic confusion patterns (e.g., 4↔9, 3↔8)
- Quantifies confidence levels for wrong predictions
- Provides actionable insights for model improvement

## 🎮 How to Use

### 1. Training Phase
```bash
python model.py
```
- Downloads MNIST dataset automatically
- Trains neural network with progress tracking
- Saves model checkpoint with metadata
- Generates training statistics and analysis

### 2. Web Interface
```bash
python predict.py
```
- **Step 1**: Wait for model to load
- **Step 2**: Click any image in the 5×5 grid
- **Step 3**: View prediction with confidence score
- **Step 4**: Use "Reset" for new random images

### 3. Analysis & Insights
```bash
python view_misidentified.py
```
- Generates comprehensive error analysis
- Creates detailed visualization grid
- Identifies model weaknesses and improvement areas

## 🔧 Configuration & Customization

### Model Hyperparameters
```python
BATCH_SIZE = 128          # Training batch size
EPOCHS = 7               # Number of training epochs
LR = 1e-3               # Learning rate
DROPOUT_RATES = [0.3, 0.2]  # Layer-wise dropout
```

### Web Interface Settings
```python
GRID_SIZE = 25           # Images per grid (5×5)
IMAGE_SIZE = 80          # Display resolution
PORT = 5000             # Flask server port
DEBUG = True            # Development mode
```

### System Optimization
```bash
# Multi-threading control
export NUM_WORKERS=4

# Device selection
export CUDA_VISIBLE_DEVICES=0
```

## 🚀 Advanced Features

### Multi-Threading Support
- Automatic CPU core detection
- Windows/Linux/macOS compatibility
- Configurable worker counts
- Memory-efficient batch loading

### GPU Acceleration
- Automatic CUDA detection
- Mixed precision training support
- Memory optimization
- Fallback to CPU mode

### Professional Error Handling
- Comprehensive exception catching
- User-friendly error messages
- Graceful degradation
- Detailed logging and debugging

## 📈 Skills Demonstrated

### 🔬 Machine Learning & AI
- Neural network architecture design
- Training pipeline implementation
- Model evaluation and validation
- Hyperparameter optimization
- Regularization techniques

### 💻 Software Engineering
- Production-ready code architecture
- Comprehensive documentation
- Error handling and logging
- Modular design patterns
- Code testing and validation

### 🌐 Web Development
- RESTful API design
- Responsive frontend development
- Real-time user interactions
- Professional UI/UX design
- Cross-platform compatibility

### 📊 Data Science & Analytics
- Statistical analysis
- Data visualization
- Pattern recognition
- Performance metrics
- Error analysis and debugging

## 🤝 Contributing

This project follows good development practices:

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes with comprehensive tests
4. **Document** all modifications
5. **Submit** a pull request with detailed description

### Code Standards
- PEP 8 Python style guidelines
- Comprehensive docstrings for all functions
- Type hints and error handling
- Unit tests for critical components
- Professional logging and debugging

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun and Corinna Cortes for the benchmark dataset
- **PyTorch**: Facebook Research for the deep learning framework
- **Flask**: Pallets team for the web framework
- **Open Source Community**: Contributors and users providing feedback

## 📞 Contact & Portfolio

### William Emeny
**Data Scientist & Machine Learning Engineer**

- **📧 Email**: williamgeorgeemeny@gmail.com
- **🐦 Twitter/X**: [@Maths_Master](https://x.com/Maths_Master)
- **📂 GitHub**: [williamemeny](https://github.com/williamemeny)
- **🌐 Portfolio**: [View my work](https://github.com/williamemeny)

### 🚀 About This Project

This project represents a comprehensive demonstration of machine learning engineering skills, from neural network design and training to web application development and deployment. It's designed to showcase competencies in:

- **Neural Network Architecture & Training**
- **Full-Stack Web Development**
- **Data Analysis & Visualization**
- **Well-Structured Software Engineering**
- **Performance Optimization & Scaling**

---

**⭐ Star this repository** if you found it helpful or inspiring!

*Built with ❤️ using PyTorch, Flask, and modern web technologies*