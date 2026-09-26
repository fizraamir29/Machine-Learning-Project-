# Real-Time Student Engagement Detection — Classical Machine Learning System

An end-to-end, classical **Machine Learning** system for real-time student engagement detection in classroom environments from live video feeds.

> **Classical Machine Learning Architecture:**
> This system replaces Deep Neural Networks (CNNs/MobileNetV2, backpropagation, and Adam) entirely with **domain-expert feature engineering** (HOG + LBP + HSV Color + Haar-like geometry) paired with scikit-learn ML classifiers: **SVM (Support Vector Machine)**, **Random Forest**, and **K-Nearest Neighbors (KNN)**.

---

## 🎯 10 Core Machine Learning Concepts Demonstrated

1. **Supervised Learning**: Model learns the mapping $f: X \to y$ from 2,277 labeled facial crops across 6 engagement states.
2. **Feature Engineering**: Transforming raw pixel matrices into an engineered 938-dimensional feature vector:
   - **HOG (Histogram of Oriented Gradients)** — 800 dimensions: Encodes facial contours, gradients, and edge geometry.
   - **Spatial LBP (Local Binary Patterns)** — 90 dimensions: Encodes micro-textures across a 3×3 facial grid.
   - **HSV Color Histograms** — 32 dimensions: Illumination-invariant skin color and tone distributions.
   - **Haar-Like Contrast Features** — 16 dimensions: Anatomical contrast differences (eyes vs forehead, mouth vs nose, symmetry).
3. **Feature Normalization (StandardScaler)**: Centers and scales all 938 dimensions to mean $\mu = 0$ and standard deviation $\sigma = 1$ to ensure equal weighting in distance/margin calculations.
4. **Support Vector Machine (SVM)**: Primary classifier maximizing margin hyperplanes with the non-linear Radial Basis Function (RBF) kernel trick ($C=10, \gamma=\text{scale}$).
5. **Random Forest (Ensemble Learning)**: Bootstrap aggregating (bagging) of 200 decorrelated decision trees, providing feature importance analysis.
6. **K-Nearest Neighbors (KNN)**: Non-parametric instance-based lazy learner using inverse distance weighting ($k=7$).
7. **Cross-Validation (5-Fold StratifiedKFold)**: Ensures balanced class distributions across folds during model validation and hyperparameter selection.
8. **Overfitting vs. Underfitting Diagnosis**: Diagnosed using Learning Curves (sample size vs. training/cross-validation accuracy).
9. **Comprehensive ML Evaluation**: Confusion Matrix, Precision, Recall, Macro/Weighted F1-score, and One-vs-Rest ROC Curves with AUC.
10. **Hyperparameter Tuning**: Systematic evaluation of SVM $C$, Random Forest $n\_estimators$, and KNN $k$ values.

---

## 📊 Dataset & Performance

- **Dataset**: 2,277 student face images across 6 classes: `Bored`, `Confused`, `Drowsy`, `Focused`, `Frustrated`, `Looking Away`
- **Data Split**: Stratified 70% Train (1,591), 15% Validation (339), 15% Held-Out Test (347)

### Model Comparison Table (Held-Out Test Set)

| Classifier | Val Accuracy | Test Accuracy | Test F1-Score | ROC AUC (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **SVM (RBF, C=10)** *(Best)* | **97.94%** | **93.08%** | **0.9306** | **0.995** |
| **Random Forest (n=200)** | 98.23% | 91.35% | 0.9134 | 0.992 |
| **KNN (k=7, distance-weighted)** | 98.23% | 90.78% | 0.9069 | 0.989 |

---

## 🛠️ Project Structure

```
├── data/
│   └── prepared/                 # Train (1591), Val (339), Test (347)
├── frontend/
│   └── index.html                # Teacher dashboard UI (HTML5/CSS3/JS)
├── models/
│   ├── best_model.joblib         # Serialized best ML model pipeline (StandardScaler + SVM)
│   ├── svm_model.joblib          # Trained SVM model
│   ├── rf_model.joblib           # Trained Random Forest model
│   ├── knn_model.joblib          # Trained KNN model
│   ├── scaler.joblib             # Fitted StandardScaler
│   └── class_names.json          # 6 target engagement class labels
├── outputs/
│   ├── classification_report.txt # Per-class Precision, Recall, F1 metrics
│   ├── confusion_matrix.png      # 6x6 test confusion matrix
│   ├── roc_curves.png            # Multi-class ROC curves (One-vs-Rest)
│   ├── learning_curves.png       # Learning curves (overfitting diagnosis)
│   ├── feature_importance.png    # Top 25 visual features (Random Forest MDI)
│   ├── model_comparison_table.txt# 3-classifier benchmark table
│   └── hyperparameter_tuning_table.txt # 5-fold CV hyperparameter results
├── src/
│   ├── app.py                    # Flask server, MJPEG video feed & REST API
│   ├── evaluate.py               # Complete ML evaluation & visualization suite
│   ├── face_detector.py          # Classical OpenCV Haar Cascade face detection
│   ├── inference.py              # Real-time feature extraction & prediction engine
│   ├── model.py                  # Hand-crafted feature engineering (HOG+LBP+Color+Haar)
│   ├── prepare_data.py           # Stratified data preparation pipeline
│   ├── train.py                  # ML model training, 5-fold CV & tuning
│   └── train_ml.py               # CLI entrypoint for training
└── requirements.txt              # Pure ML dependencies (No TensorFlow)
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Machine Learning Models
Extract hand-crafted visual features, perform 5-fold cross-validation, and train SVM, Random Forest, and KNN:
```bash
python src/train_ml.py
```
*(or `python src/train.py`)*

### 3. Evaluate & Generate Diagnostic Plots
Generate confusion matrix, ROC curves, learning curves, and feature importance:
```bash
python src/evaluate.py
```

### 4. Launch Real-Time Dashboard
Start the Flask backend and live engagement detection interface:
```bash
python src/app.py
```
Open **`http://localhost:5000`** in your browser.
