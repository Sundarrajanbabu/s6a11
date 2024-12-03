# MNIST Classification with CI/CD Pipeline

This project implements a CNN-based MNIST digit classifier with a complete CI/CD pipeline using GitHub Actions. The model achieves 99.26% validation accuracy while maintaining under 20K parameters.

## Model Architecture

- **Input**: 28x28 grayscale images
- **Architecture**: 3-block CNN with:
  - Progressive channel expansion (8 → 16 → 32)
  - BatchNorm after convolutions
  - Progressive dropout (0.05 → 0.1 → 0.25)
  - MaxPooling strategically placed
  - Final FC layer for classification

### Key Features
- Total Parameters: < 20K
- Uses Batch Normalization
- Implements Dropout for regularization
- Includes Fully Connected layer
- Optimized pooling placement

## Training Details

- **Dataset Split**:
  - Training: 60,000 images (MNIST training set)
  - Validation: 10,000 images (MNIST test set)

- **Training Configuration**:
  - Epochs: 19
  - Batch Size: 64
  - Optimizer: SGD
    - Initial LR: 0.1
    - Momentum: 0.9
    - Weight Decay: 1e-4
  
- **Learning Rate Schedule**:
  - Epoch 8: 0.05
  - Epoch 12: 0.01
  - Epoch 15: 0.005

- **Data Augmentation**:
  - Random Affine (±5°)
  - Random Translation (±10%)
  - Random Scaling (90-110%)
  - Random Erasing (p=0.1)

## Performance

- Best Validation Accuracy: 99.26%
- Target Accuracy: 99.40%
- Consistent performance across runs

## CI/CD Pipeline

The GitHub Actions pipeline performs:
1. Model Training
2. Validation Accuracy Check
3. Architecture Validation:
   - Parameter Count (< 20K)
   - BatchNorm Presence
   - Dropout Implementation
   - FC Layer Presence
4. Model Artifact Storage

## Requirements 