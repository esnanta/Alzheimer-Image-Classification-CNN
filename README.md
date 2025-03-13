# Alzheimer Image Classification using Convolutional Neural Network (CNN)

## 📌 About Alzheimer’s Disease
Alzheimer's disease is a progressive neurological disorder that leads to memory loss and cognitive decline. It is one of the most common causes of dementia, affecting millions of people worldwide. Early detection through MRI imaging can assist in diagnosis and treatment.

![Sentiment Analysis Word Clouds](https://github.com/esnanta/alzheimer-image-classification-cnn/blob/53ace85414cdc71ec077d7ca77bf02f5daef81ee/image/alzheimer_images.png?raw=true)

## 🧠 About the Dataset
This dataset contains MRI images categorized into four classes:
- **Mild Demented**: 8,960 images
- **Moderate Demented**: 6,464 images
- **Very Mild Demented**: 8,960 images
- **Non Demented**: 9,600 images

📌 **Dataset Link**: [Alzheimer Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)

## 🏗 Project Structure
This project classifies Alzheimer’s disease using CNN. Below is the structured workflow:

### 1️⃣ Importing Required Libraries

### 2️⃣ Data Preparation

### 3️⃣ Dataset Splitting
The dataset is divided into:
- **70% Training**
- **20% Validation**
- **10% Testing**

### 4️⃣ Data Loading

### 5️⃣ Preprocessing Data
- **Resize Images** to 224x224 pixels for uniform input size.
- **Normalize Pixel Values** to improve model stability.
- **Shuffle Data** to prevent order bias during training.
- **Batching & Prefetching** to optimize training speed.

### 6️⃣ Model Architecture
The CNN model consists of:
- **Feature Extraction**
  - `Conv2D` + `BatchNormalization` + `ReLU` Activation (Extracts features from MRI images)
  - `MaxPooling2D` (Reduces feature dimension and prevents overfitting)
  - **5 Convolution Layers** to capture complex patterns
- **Classification Layers**
  - `Dense(128) + BatchNormalization + ReLU` (Fully connected layer for feature representation)
  - `Dropout(0.5)` (Reduces overfitting)
  - `Dense(4, activation='softmax')` (Outputs probability for 4 classes)
- **Training Setup**
  - `Adam Optimizer` (Adaptive learning for stable convergence)
  - `Sparse Categorical Crossentropy` (Handles integer class labels efficiently)
  - **Callbacks:**
    - `EarlyStopping`: Stops training if no improvement or accuracy reaches **≥95%**.
    - `ReduceLROnPlateau`: Reduces learning rate if validation loss stagnates.
    - **Custom Callback:** Stops training when validation accuracy hits **95%**.

### 7️⃣ Evaluation and Visualization
After training, the model achieves:
- **Test Accuracy**: **96.59%**
- **Test Loss**: **0.0991**

#### Training & Validation Accuracy Analysis
📈 **Accuracy Trends:**
- Sharp accuracy increase at the beginning, stabilizing around **95%** for validation accuracy.
- Training accuracy almost reaches **100%**, with validation accuracy around **95%**.
- Minimal gap between training and validation accuracy, indicating **good generalization**.

#### Training & Validation Loss Analysis
📉 **Loss Trends:**
- Training loss consistently decreases, showing the model learns well.
- Validation loss also decreases but fluctuates slightly mid-training.
- The small gap between training and validation loss confirms **no significant overfitting**.

## 📢 Conclusion
This CNN-based Alzheimer image classification model demonstrates high accuracy and stability in detecting different stages of Alzheimer's disease using MRI images. The balanced preprocessing, proper augmentations, and deep network architecture contribute to its efficiency and generalization.

🚀 **Next Steps:**
- Improve model robustness with more data augmentation.
- Experiment with transfer learning (e.g., ResNet, EfficientNet).
- Optimize hyperparameters for further accuracy gains.

📌 **Developed with TensorFlow & Keras.**
