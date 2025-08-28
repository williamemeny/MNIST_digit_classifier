// =============================================================================
// MNIST Test Image Predictor - Interactive JavaScript Application
// =============================================================================
// 
// This JavaScript file provides all the interactive functionality for the
// MNIST test image prediction interface, including:
//
// - Model status checking and display
// - Dynamic image grid generation and management
// - Image selection and prediction handling
// - Real-time UI updates and feedback
// - Error handling and user notifications
//
// =============================================================================

// =============================================================================
// GLOBAL VARIABLES AND STATE MANAGEMENT
// =============================================================================

let currentSelection = null;
let currentImages = [];
let modelStatus = {
    loaded: false,
    message: '',
    info: {}
};

// =============================================================================
// DOM ELEMENT REFERENCES
// =============================================================================

const elements = {
    // Grid elements
    imageGrid: document.getElementById('imageGrid'),
    resetBtn: document.getElementById('resetBtn'),
    
    // Model status elements
    modelStatusCard: document.getElementById('modelStatusCard'),
    statusIndicator: document.getElementById('statusIndicator'),
    modelInfoCard: document.getElementById('modelInfoCard'),
    
    // Model info elements
    modelArchitecture: document.getElementById('modelArchitecture'),
    modelParameters: document.getElementById('modelParameters'),
    testAccuracy: document.getElementById('testAccuracy'),
    bestAccuracy: document.getElementById('bestAccuracy'),
    modelDevice: document.getElementById('modelDevice'),
    
    // Selection elements
    selectionCard: document.getElementById('selectionCard'),
    selectedImage: document.getElementById('selectedImage'),
    actualDigit: document.getElementById('actualDigit'),
    gridPosition: document.getElementById('gridPosition'),
    predictBtn: document.getElementById('predictBtn'),
    
    // Results elements
    resultsCard: document.getElementById('resultsCard'),
    predictionResult: document.getElementById('predictionResult')
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function showNotification(message, type = 'info') {
    /**
     * Display a temporary notification message to the user.
     * 
     * @param {string} message - The message to display
     * @param {string} type - The type of notification ('info', 'success', 'error')
     */
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Show with animation
    setTimeout(() => notification.classList.add('show'), 100);
    
    // Remove after delay
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function formatNumber(num) {
    /**
     * Format large numbers with commas for readability.
     * 
     * @param {number} num - The number to format
     * @returns {string} Formatted number string
     */
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function formatPercentage(value) {
    /**
     * Format decimal values as percentages.
     * 
     * @param {number} value - The decimal value (0-1)
     * @returns {string} Formatted percentage string
     */
    return `${(value * 100).toFixed(1)}%`;
}

// =============================================================================
// MODEL STATUS MANAGEMENT
// =============================================================================

async function checkModelStatus() {
    /**
     * Check the current status of the loaded model.
     */
    try {
        const response = await fetch('/api/model/status');
        const data = await response.json();
        
        modelStatus = data;
        updateModelStatusDisplay();
        
        if (data.loaded) {
            // Model is ready, load test images
            await loadRandomImages();
        }
        
    } catch (error) {
        console.error('Failed to check model status:', error);
        showNotification('Failed to check model status', 'error');
        updateModelStatusDisplay();
    }
}

function updateModelStatusDisplay() {
    /**
     * Update the UI to reflect the current model status.
     */
    const statusIndicator = elements.statusIndicator;
    
    if (modelStatus.loaded) {
        // Model is loaded and ready
        statusIndicator.innerHTML = `
            <span style="color: #059669;">✅</span>
            <span>${modelStatus.message}</span>
        `;
        statusIndicator.className = 'status-indicator connected';
        
        // Show model info if available
        if (modelStatus.info && Object.keys(modelStatus.info).length > 0) {
            showModelInfo();
        }
        
    } else {
        // Model is not loaded
        statusIndicator.innerHTML = `
            <span style="color: #dc2626;">❌</span>
            <span>${modelStatus.message || 'Model not available'}</span>
        `;
        statusIndicator.className = 'status-indicator error';
        
        // Hide model info
        elements.modelInfoCard.style.display = 'none';
    }
}

function showModelInfo() {
    /**
     * Display detailed model information in the UI.
     */
    if (!modelStatus.info || Object.keys(modelStatus.info).length === 0) {
        return;
    }
    
    const info = modelStatus.info;
    
    // Update model info display
    elements.modelArchitecture.textContent = info.architecture || 'N/A';
    elements.modelParameters.textContent = info.total_parameters ? formatNumber(info.total_parameters) : 'N/A';
    elements.testAccuracy.textContent = info.test_accuracy !== 'N/A' ? formatPercentage(info.test_accuracy) : 'N/A';
    elements.bestAccuracy.textContent = info.best_test_accuracy !== 'N/A' ? formatPercentage(info.best_test_accuracy) : 'N/A';
    elements.modelDevice.textContent = info.device || 'N/A';
    
    // Show the model info card
    elements.modelInfoCard.style.display = 'block';
}

// =============================================================================
// IMAGE GRID MANAGEMENT
// =============================================================================

async function loadRandomImages() {
    /**
     * Load and display a new set of random test images.
     */
    try {
        // Show loading state
        elements.imageGrid.innerHTML = `
            <div class="loading-placeholder">
                <div class="spinner"></div>
                <p>Loading test images...</p>
            </div>
        `;
        
        // Fetch random images from server
        const response = await fetch('/api/images/random');
        const data = await response.json();
        
        if (data.success) {
            currentImages = data.images;
            renderImageGrid();
            elements.resetBtn.disabled = false;
        } else {
            throw new Error(data.error || 'Failed to load images');
        }
        
    } catch (error) {
        console.error('Failed to load random images:', error);
        showNotification('Failed to load test images', 'error');
        
        // Show error state
        elements.imageGrid.innerHTML = `
            <div class="loading-placeholder">
                <span style="color: #dc2626; font-size: 2rem;">❌</span>
                <p>Failed to load images</p>
                <button class="btn btn-primary" onclick="loadRandomImages()">Retry</button>
            </div>
        `;
    }
}

function renderImageGrid() {
    /**
     * Render the current set of images in the 5x5 grid.
     */
    if (!currentImages || currentImages.length === 0) {
        return;
    }
    
    const gridHTML = currentImages.map((imageData, index) => {
        const row = Math.floor(index / 5) + 1;
        const col = (index % 5) + 1;
        
        return `
            <div class="grid-item" data-index="${index}" onclick="selectImage(${index})">
                <img src="${imageData.image}" alt="MNIST digit ${imageData.label}">
                <div class="image-label">${imageData.label}</div>
            </div>
        `;
    }).join('');
    
    elements.imageGrid.innerHTML = gridHTML;
}

function selectImage(index) {
    /**
     * Handle image selection in the grid.
     * 
     * @param {number} index - The index of the selected image
     */
    if (index < 0 || index >= currentImages.length) {
        return;
    }
    
    // Clear previous selection
    const previousSelected = elements.imageGrid.querySelector('.grid-item.selected');
    if (previousSelected) {
        previousSelected.classList.remove('selected');
    }
    
    // Mark new selection
    const selectedItem = elements.imageGrid.querySelector(`[data-index="${index}"]`);
    if (selectedItem) {
        selectedItem.classList.add('selected');
    }
    
    // Store selection data
    const imageData = currentImages[index];
    currentSelection = {
        index: index,
        imageData: imageData
    };
    
    // Update selection display
    updateSelectionDisplay();
    
    // Show selection card
    elements.selectionCard.style.display = 'block';
    
    // Hide results card
    elements.resultsCard.style.display = 'none';
}

function updateSelectionDisplay() {
    /**
     * Update the selection display with the currently selected image.
     */
    if (!currentSelection) {
        return;
    }
    
    const imageData = currentSelection.imageData;
    const row = Math.floor(currentSelection.index / 5) + 1;
    const col = (currentSelection.index % 5) + 1;
    
    // Update image display
    elements.selectedImage.src = imageData.image;
    elements.actualDigit.textContent = imageData.label;
    elements.gridPosition.textContent = `Row ${row}, Column ${col}`;
}

async function resetImageGrid() {
    /**
     * Reset the image grid with new random images.
     */
    // Clear current selection
    currentSelection = null;
    elements.selectionCard.style.display = 'none';
    elements.resultsCard.style.display = 'none';
    
    // Load new random images
    await loadRandomImages();
}

// =============================================================================
// PREDICTION HANDLING
// =============================================================================

async function predictSelectedImage() {
    /**
     * Make a prediction on the currently selected image.
     */
    if (!currentSelection) {
        showNotification('Please select an image first', 'error');
        return;
    }
    
    if (!modelStatus.loaded) {
        showNotification('Model is not loaded', 'error');
        return;
    }
    
    try {
        // Disable predict button during prediction
        elements.predictBtn.disabled = true;
        elements.predictBtn.textContent = '🔄 Predicting...';
        
        // Send image data to server for prediction
        const response = await fetch('/api/images/select', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_index: currentSelection.index,
                image_tensor: currentSelection.imageData.tensor || [],
                label: currentSelection.imageData.label
            })
        });
        
        const selectData = await response.json();
        
        if (!selectData.success) {
            throw new Error(selectData.error || 'Failed to select image');
        }
        
        // Make prediction
        const predictResponse = await fetch('/api/predict', {
            method: 'POST'
        });
        
        const predictData = await predictResponse.json();
        
        if (predictData.success) {
            displayPredictionResults(predictData);
        } else {
            throw new Error(predictData.error || 'Prediction failed');
        }
        
    } catch (error) {
        console.error('Prediction failed:', error);
        showNotification(`Prediction failed: ${error.message}`, 'error');
    } finally {
        // Re-enable predict button
        elements.predictBtn.disabled = false;
        elements.predictBtn.textContent = '🚀 Predict Digit';
    }
}

function displayPredictionResults(results) {
    /**
     * Display the prediction results in the UI.
     * 
     * @param {Object} results - The prediction results object
     */
    const isCorrect = results.is_correct;
    const predictedDigit = results.predicted_digit;
    const actualDigit = results.actual_digit;
    const confidence = results.confidence;
    const confidencePercentage = results.confidence_percentage;
    
    // Create results HTML
    const resultsHTML = `
        <div class="prediction-success ${isCorrect ? 'correct' : 'incorrect'}">
            <div class="prediction-digit">${predictedDigit}</div>
            <p><strong>Predicted Digit:</strong> ${predictedDigit}</p>
            <p><strong>Actual Digit:</strong> ${actualDigit}</p>
            <p><strong>Confidence:</strong> ${confidencePercentage}</p>
            
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
            </div>
            
            <p style="margin-top: 1rem;">
                ${isCorrect ? '✅ Correct prediction!' : '❌ Incorrect prediction'}
            </p>
        </div>
    `;
    
    // Update results display
    elements.predictionResult.innerHTML = resultsHTML;
    
    // Show results card
    elements.resultsCard.style.display = 'block';
    
    // Show success/error notification
    if (isCorrect) {
        showNotification(`Correct! Predicted ${predictedDigit} with ${confidencePercentage} confidence`, 'success');
    } else {
        showNotification(`Incorrect. Predicted ${predictedDigit} but actual was ${actualDigit}`, 'error');
    }
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

function setupEventListeners() {
    /**
     * Set up all event listeners for the application.
     */
    
    // Reset button
    elements.resetBtn.addEventListener('click', resetImageGrid);
    
    // Predict button
    elements.predictBtn.addEventListener('click', predictSelectedImage);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (event) => {
        if (event.key === 'r' || event.key === 'R') {
            // R key to reset grid
            if (!elements.resetBtn.disabled) {
                resetImageGrid();
            }
        } else if (event.key === 'Enter') {
            // Enter key to predict
            if (currentSelection && !elements.predictBtn.disabled) {
                predictSelectedImage();
            }
        }
    });
}

// =============================================================================
// APPLICATION INITIALIZATION
// =============================================================================

async function initializeApplication() {
    /**
     * Initialize the application when the page loads.
     */
    try {
        console.log('🚀 Initializing MNIST Test Image Predictor...');
        
        // Set up event listeners
        setupEventListeners();
        
        // Check model status
        await checkModelStatus();
        
        console.log('✅ Application initialized successfully');
        
    } catch (error) {
        console.error('❌ Failed to initialize application:', error);
        showNotification('Failed to initialize application', 'error');
    }
}

// =============================================================================
// PAGE LOAD EVENT
// =============================================================================

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApplication);
} else {
    initializeApplication();
}

// =============================================================================
// EXPORT FOR TESTING (if needed)
// =============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        checkModelStatus,
        loadRandomImages,
        selectImage,
        predictSelectedImage,
        resetImageGrid
    };
}

