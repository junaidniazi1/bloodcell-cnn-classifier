# 🔬 Automated Blood Cell Classification for Leukemia Detection

A deep learning project to classify microscopic blood cell images for leukemia detection using CNN architecture.

## 📊 Project Overview

This project uses images sourced from **Kaggle** to classify blood cells into five categories for early leukemia detection.

### Blood Cell Categories
- Basophils
- Erythroblasts
- Monocytes
- Myeloblasts
- Segmented Neutrophils

## 🔄 Project Workflow

1. **Data Collection**: High-resolution blood cell images from Kaggle
2. **Data Preprocessing**: Split dataset into train, validation, and test sets
3. **Data Augmentation**: Applied flipping, rotation, zoom, and brightness adjustments
4. **Model Training**: Built and trained CNN architecture
5. **Deployment**: Created interactive Streamlit dashboard

## 🧠 Model Architecture

**CNN** with the following layers:
- Conv2D layers
- Batch Normalization
- Dropout
- MaxPooling

**Training Configuration:**
- Optimizer: Adam
- Loss Function: Categorical Crossentropy

## 📈 Model Performance

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 99.95% | 96.49% | 96.09% |
| **Loss** | 0.3456 | 0.4370 | 0.4433 |

## 🚀 Deployment

Interactive **Streamlit Dashboard** featuring:
- Real-time image upload
- Blood cell type prediction
- Confidence scores
- Probability distribution for all classes

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: NumPy, PIL
- **Visualization**: Matplotlib
- **Deployment**: Streamlit

## 💡 Use Case

This project demonstrates the application of **AI and deep learning in medical diagnostics**, providing a robust tool for early leukemia detection.

---

⚕️ *For research and educational purposes only. Not a substitute for professional medical diagnosis.*
