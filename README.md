# Early Diagnosis and Grading of Age-Related Macular Degeneration (AMD) using Deep Learning

![Project Banner](https://github.com/RadhaVaishnavi/AMD/assets/84319477/b56b2a6d-8f35-4c9a-ac4e-13434b55e8a2)

## Overview

This project aims to develop a deep learning-based system for the **early detection and classification** of retinal diseases from Optical Coherence Tomography (OCT) images. 

The model classifies OCT scans into four categories:
- **CNV** (Choroidal Neovascularization)
- **DME** (Diabetic Macular Edema)
- **DRUSEN**
- **NORMAL**

We explored multiple convolutional neural network architectures and fine-tuned several pretrained models. After extensive experimentation, **ResNet50** emerged as the best-performing model with **93.9% accuracy** on the test set.

Early and accurate diagnosis of AMD and related conditions is critical to prevent irreversible vision loss. This work demonstrates the potential of transfer learning in medical image analysis.

## Dataset

We used the **Retinal OCT Images** dataset (Kermany et al., 2018).

- **Total Images**: 84,495 JPEG images
- **Classes**: CNV, DME, DRUSEN, NORMAL
- **Source**: Collected from multiple clinical centers (UC San Diego, Shanghai First People’s Hospital, etc.)
- **Labeling**: Multi-tier verification by ophthalmologists and retinal specialists

### Dataset Split
- 80% Training
- 10% Validation  
- 10% Testing (968 images used for final evaluation)

### Preprocessing Steps
- Resized all images to **224 × 224** pixels
- Normalization using ImageNet mean and std
- One-hot encoding of class labels
- Data augmentation (random rotation, horizontal flip, zoom, brightness adjustment)
- Experiments conducted on both balanced and imbalanced class distributions

## Methodology

### 1. Exploratory Analysis
- RGB channel-wise visualization of OCT images
- Analysis of image dimensions and class distribution

### 2. Layer Segmentation (Exploratory)
We experimented with **CE-Net** using modified loss functions (Dice, Tversky, Weighted Binary Cross-Entropy, and Regularization) for retinal layer segmentation. This step helps in understanding structural changes but was not used in the final classification pipeline.

### 3. Proposed Model: Fine-tuned ResNet50

**ResNet50** (pretrained on ImageNet) was selected as the best model due to its residual connections that facilitate effective feature learning for subtle retinal patterns.

**Model Architecture**:
- Pretrained ResNet50 backbone
- Global Average Pooling layer
- Dropout (rate = 0.5) for regularization
- Dense layer with ReLU activation
- Softmax output layer for 4-class classification

**Training Details**:
- Optimizer: Adam (learning rate = 0.001 with decay)
- Loss: Categorical Cross-Entropy
- Batch size: 32
- Epochs: Up to 50 with early stopping
- Input size: 224 × 224 × 3

We also implemented and compared the following models:
- VGG-16
- InceptionV3
- Custom CNN architectures

## Results

### Performance Comparison

| Model                  | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|------------------------|--------------|---------------|------------|--------------|
| **ResNet50 (Proposed)**    | **93.9**    | 93.7         | 93.8      | 93.7        |
| InceptionV3            | 91.8        | 91.5         | 91.6      | 91.5        |
| VGG-16                 | 90.2        | 89.9         | 90.1      | 90.0        |
| Custom CNN             | 86.5        | 86.2         | 86.4      | 86.3        |

The proposed **ResNet50** model achieved the highest performance with **93.9% accuracy**, outperforming other architectures while maintaining good generalization.

### Additional Metrics
- High True Positive Rate (TPR) and True Negative Rate (TNR) across all classes
- Balanced performance even on the minority classes

**Visualizations** (included in the repository):
- Training and validation accuracy/loss curves
- Confusion matrix
- Sample predictions (correct and incorrect)
- Grad-CAM visualizations for model interpretability

## Comparative Discussion

ResNet50 outperformed the other models thanks to:
- Effective transfer learning from ImageNet
- Residual connections that help learn deeper representations
- Proper regularization (Dropout + data augmentation)
- Careful hyperparameter tuning

The model shows promising results for assisting ophthalmologists in early screening of AMD-related conditions.

