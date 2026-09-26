# 📘 Student Engagement Detection — Comprehensive Machine Learning Project Documentation & Viva Defense Guide

> **Project Title:** Real-Time Student Engagement Detection in Classroom using Classical Machine Learning  
> **Student / Author:** Fizra Amir (`fizraamir29`)  
> **Repository:** [https://github.com/fizraamir29/Machine-Learning-Project-](https://github.com/fizraamir29/Machine-Learning-Project-)  
> **Target Problem:** Real-time multi-face classification of student engagement states (`Bored`, `Confused`, `Drowsy`, `Focused`, `Frustrated`, `Looking Away`) from live classroom camera feeds.

---

## 📑 Table of Contents
1. [Executive Overview (Project Story & Architecture)](#1-executive-overview)
2. [Dataset & Data Preparation (`prepare_data.py`)](#2-dataset--data-preparation)
3. [File-by-File Detailed Walkthrough](#3-file-by-file-detailed-walkthrough)
   - [File 1: `src/model.py` (Feature Engineering Engine)](#file-1-srcmodelpy)
   - [File 2: `src/train.py` & `src/train_ml.py` (Model Training & Cross-Validation)](#file-2-srctrainpy--srctrain_mlpy)
   - [File 3: `src/evaluate.py` (Model Evaluation & Diagnostic Curves)](#file-3-srcevaluatepy)
   - [File 4: `src/face_detector.py` (OpenCV Haar Cascade Face Detector)](#file-4-srcface_detectorpy)
   - [File 5: `src/inference.py` (Real-Time ML Inference Engine)](#file-5-srcinferencepy)
   - [File 6: `src/app.py` (Flask Backend, MJPEG Stream & REST API)](#file-6-srcapppy)
   - [File 7: `frontend/index.html` (Teacher Dashboard UI)](#file-7-frontendindexhtml)
   - [File 8: `requirements.txt` (Dependencies Analysis)](#file-8-requirementstxt)
4. [Mastery of the 10 Core Machine Learning Concepts](#4-mastery-of-the-10-core-machine-learning-concepts)
5. [Model Benchmark & Experimental Results](#5-model-benchmark--experimental-results)
6. [Teacher Viva / Defense Q&A (Urdu + English)](#6-teacher-viva--defense-qa-urdu--english)

---

## 1. Executive Overview

### Why Classical Machine Learning Instead of Deep Learning?
In traditional computer vision, Deep Learning (CNNs like MobileNetV2, ResNet) learns features implicitly via millions of parameters. However:
1. **Explainability & Transparency:** In academic ML, a neural network is a "black box". Classical ML allows us to explicitly define and extract mathematical visual features (gradients, textures, skin tones, facial symmetry).
2. **Computational Efficiency:** Deep learning requires heavy GPUs or high CPU inference latency. Classical ML runs feature extraction + SVM inference in **sub-millisecond speed** (~3 ms per face crop), enabling real-time multi-face classroom tracking on any standard laptop.
3. **Data Efficiency & Regularization:** On moderate datasets (~2,000 images), deep CNNs are prone to overfitting or batch-norm running statistic collapse. Margin-maximizing classifiers like SVM generalize with high test accuracy (**93.08%**).

### High-Level Architecture Flowchart

```
[ Webcam / Video Stream ]
           │
           ▼
[ OpenCV Haar Cascade Detection ]  ───> Detects bounding boxes (x, y, w, h)
           │
           ▼
[ Face Crop Normalization (96x96) ]
           │
           ▼
[ Feature Engineering (model.py) ]
   ├── 1. HOG Features           (800 dims)  ──> Shape, edge orientations
   ├── 2. Spatial LBP Features   ( 90 dims)  ──> Micro-textures (eyes, skin)
   ├── 3. HSV Color Histograms   ( 32 dims)  ──> Illumination-invariant color
   └── 4. Haar-like Differences  ( 16 dims)  ──> Facial landmark contrasts
           │
           ▼ Concatenate
[ 938-Dimensional Feature Vector ]
           │
           ▼
[ StandardScaler Normalization ]   ───> (x - mean) / std (Zero-mean, unit variance)
           │
           ▼
[ Trained Classifiers (SVM / RF / KNN) ]
           │
           ▼
[ Class Probability Distribution & Final Label ]
           │
           ▼
[ Flask MJPEG Stream + Real-Time Teacher Analytics Dashboard ]
```

---

## 2. Dataset & Data Preparation

### File: `src/prepare_data.py`

#### What this file does:
Merges 3 distinct raw datasets into a standardized, balanced 6-class dataset and performs a stratified 70/15/15 split.

- **Kaggle Dataset:** `Student-engagement-dataset` (2,120 raw images)
- **Maleha Dataset:** Custom cropped classroom images (32 images)
- **Zoha Dataset:** Mobile-captured classroom images (125 images, auto-converted HEIC format to JPG)
- **Total Unified Dataset:** **2,277 verified face images**

#### Data Splits:
- **Train Set (70%):** 1,591 images (used for model fitting and 5-fold cross-validation)
- **Validation Set (15%):** 339 images (used for intermediate validation and model selection)
- **Test Set (15%):** 347 images (strictly held-out, used ONLY for final unbiased evaluation)

#### Target Engagement Classes (6):
1. `Bored`
2. `Confused`
3. `Drowsy`
4. `Focused`
5. `Frustrated`
6. `Looking Away`

#### ML Concept Applied:
- **Stratified Sampling:** Ensures that every split (train, val, test) contains the exact same percentage proportion of each class as the original dataset. This prevents class imbalance and data skew.

---

## 3. File-by-File Detailed Walkthrough

---

### FILE 1: `src/model.py`
**Purpose:** Hand-Crafted Feature Engineering Module (The Core ML Innovation).

#### Key Functions & Logic:
1. `extract_hog_features(gray_img)`:
   - Uses `skimage.feature.hog`.
   - Divides 96×96 face into 16×16 pixel cells with 2×2 cells per block and 8 gradient orientation bins.
   - Outputs: $5 \times 5 \times 4 \times 8 = \mathbf{800\text{ dimensions}}$.
   - *Why?* Captures head angle, eyebrow arches, eye shapes, and jaw contours.
2. `extract_lbp_features(gray_img, grid_size=(3, 3))`:
   - Uses `skimage.feature.local_binary_pattern` ($P=8, R=1$, uniform pattern).
   - Uniform LBP creates 10 histogram bins (0–9).
   - Computes local histograms across a 3×3 spatial grid of the face ($3 \times 3 \times 10 = \mathbf{90\text{ dimensions}}$).
   - *Why?* Distinguishes smooth skin from forehead frown wrinkles, open eye texture from closed eye lids.
3. `extract_color_features(bgr_img)`:
   - Converts BGR to HSV color space.
   - Extracts 16 Hue bins, 8 Saturation bins, 8 Value bins ($\mathbf{32\text{ dimensions}}$).
   - *Why?* Hue separates skin color from shadows and classroom lighting changes.
4. `extract_haar_like_features(gray_img)`:
   - Calculates relative intensity differences across face zones:
     - Forehead vs. Eyes (eyebrow furrowing / squinting proxy)
     - Nose vs. Mouth (mouth opening / yawning proxy)
     - Left Half vs. Right Half (head turn / looking away proxy)
     - 4-quadrant rectangular Haar contrast differences ($\mathbf{16\text{ dimensions}}$).
5. `extract_features(bgr_crop)`:
   - Concatenates: $800\text{ (HOG)} + 90\text{ (LBP)} + 32\text{ (Color)} + 16\text{ (Haar)} = \mathbf{938\text{ total dimensions}}$.
6. `get_feature_names()`:
   - Generates exact semantic names for all 938 dimensions for model interpretability.

#### Libraries Used:
- `cv2` (OpenCV): Color conversions (BGR $\to$ Grayscale, BGR $\to$ HSV), image resizing.
- `skimage.feature` (scikit-image): `hog`, `local_binary_pattern`.
- `numpy`: Vector stacking, array manipulation.

---

### FILE 2: `src/train.py` & `src/train_ml.py`
**Purpose:** Classical Model Training, 5-Fold Stratified Cross-Validation & Hyperparameter Tuning.

#### Key Functions & Logic:
1. `load_dataset_features(split_dir, class_names, cache_file)`:
   - Iterates through the directory, runs `extract_features()` for each image, and caches extracted features to `.npz` files for high-speed repeatability.
2. `create_model_pipelines()`:
   - Bundles feature preprocessing (`StandardScaler`) with classifiers in `sklearn.pipeline.Pipeline`:
     - **SVM Pipeline:** `StandardScaler() + SVC(C=10.0, kernel='rbf', gamma='scale', probability=True)`
     - **Random Forest Pipeline:** `StandardScaler() + RandomForestClassifier(n_estimators=200, n_jobs=-1)`
     - **KNN Pipeline:** `StandardScaler() + KNeighborsClassifier(n_neighbors=7, weights='distance')`
3. `perform_hyperparameter_study(X, y)`:
   - Executes 5-fold cross-validation across:
     - SVM $C \in [0.1, 1.0, 10.0, 50.0]$
     - Random Forest $n\_estimators \in [50, 100, 200]$
     - KNN $k \in [3, 5, 7, 9]$
   - Generates a formatted ASCII comparison table saved to `outputs/hyperparameter_tuning_table.txt`.
4. `main()`:
   - Trains all 3 classifiers on Train + Val data ($1,930$ samples).
   - Evaluates all 3 models on the held-out Test set ($347$ samples).
   - Saves the champion model (`best_model.joblib`), individual models (`svm_model.joblib`, `rf_model.joblib`, `knn_model.joblib`), scaler (`scaler.joblib`), and `class_names.json`.

#### Libraries Used:
- `sklearn.svm.SVC`: Support Vector Classification.
- `sklearn.ensemble.RandomForestClassifier`: Random Forest ensemble.
- `sklearn.neighbors.KNeighborsClassifier`: KNN algorithm.
- `sklearn.preprocessing.StandardScaler`: Feature normalization.
- `sklearn.model_selection.StratifiedKFold`, `cross_val_score`: Cross-validation.
- `joblib`: Model serialization.

---

### FILE 3: `src/evaluate.py`
**Purpose:** Comprehensive Diagnostic & Evaluation Suite.

#### Key Functions & Logic:
1. `evaluate_confusion_matrix(model, X_test, y_test, class_names, out_dir)`:
   - Computes $6 \times 6$ confusion matrix and classification report.
   - Plots and saves `outputs/confusion_matrix.png` and `outputs/classification_report.txt`.
2. `plot_roc_curves(model, X_test, y_test, class_names, out_dir)`:
   - Implements One-vs-Rest (OvR) multi-class Receiver Operating Characteristic curves.
   - Computes False Positive Rate (FPR), True Positive Rate (TPR), and Area Under the Curve (AUC) for each of the 6 classes.
   - Saves `outputs/roc_curves.png`.
3. `plot_learning_curves(model, X, y, out_dir)`:
   - Uses `sklearn.model_selection.learning_curve` across training sample sizes ($20\%$ to $100\%$).
   - Plots Training Accuracy vs. Cross-Validation Accuracy with shaded variance bands.
   - Diagnoses generalization gap (proves absence of high overfitting or underfitting).
   - Saves `outputs/learning_curves.png`.
4. `plot_feature_importance(rf_model, out_dir, top_n=25)`:
   - Extracts Gini impurity reduction from the trained Random Forest.
   - Matches indices with `get_feature_names()`.
   - Plots horizontal bar chart of the top 25 most critical features in `outputs/feature_importance.png`.
5. `compare_all_classifiers(models, X_test, y_test, class_names, out_dir)`:
   - Benchmarks SVM vs. RF vs. KNN on Accuracy, Precision, Recall, F1-Score, and ROC AUC.
   - Saves `outputs/model_comparison_table.txt`.

---

### FILE 4: `src/face_detector.py`
**Purpose:** OpenCV Haar Cascade Face Detection (Classical ML).

#### Key Functions & Logic:
1. `_face_cascade`:
   - Loads OpenCV's pre-trained Viola-Jones Haar Cascade XML: `haarcascade_frontalface_default.xml`.
2. `detect_faces(frame_bgr, min_size=(80, 80))`:
   - Converts video frame to grayscale.
   - Applies histogram equalization (`cv2.equalizeHist`) to handle low/high room illumination.
   - Runs `detectMultiScale(scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))`.
   - Returns list of bounding boxes $[(x, y, w, h), \dots]$.
   - Includes intelligent fallback centering in case of complete camera occlusion.
3. `crop_face(frame_bgr, box, margin=0.2)`:
   - Adds a 20% margin around the detected box to capture hair, chin, and forehead context without clipping.

---

### FILE 5: `src/inference.py`
**Purpose:** Real-Time Feature Extraction and Model Prediction Engine.

#### Key Functions & Logic:
1. `EngagementPredictor.__init__(model_path, class_names_path)`:
   - Loads the serialized `best_model.joblib` pipeline and class dictionary.
2. `predict(face_bgr)`:
   - Predicts single face crop.
3. `predict_batch(face_crops)`:
   - Loops over all faces detected in the current camera frame.
   - Extracts 938-dim feature vectors for each face using `model.extract_features()`.
   - Calls `model.predict_proba()` from the scikit-learn pipeline.
   - Formats outputs as: `(predicted_label, confidence_score, all_class_probabilities)`.
4. `_fallback_heuristic_batch(face_crops)`:
   - Rule-based facial intensity heuristic that acts as a failsafe if no model is loaded.

---

### FILE 6: `src/app.py`
**Purpose:** Flask Server, MJPEG Camera Streaming, and Session Analytics REST API.

#### Key Functions & Logic:
1. `SessionState`:
   - Thread-safe class tracking active classroom sessions, elapsed time, running engagement scores, and per-class counts.
   - Engaged categories defined as: `Focused + Confused + Frustrated` (active participation).
   - Generates automated pedagogical feedback (e.g., advising a 2-minute energizer break if drowsiness exceeds 30%).
2. `gen_frames()`:
   - Captures frames from `cv2.VideoCapture(0)`.
   - Detects faces with `detect_faces()`.
   - Predicts engagement using `EngagementPredictor.predict_batch()`.
   - Draws color-coded bounding boxes and badges on each face in OpenCV BGR:
     - 🟢 Lime Green: `Focused`
     - 🟡 Amber Yellow: `Confused`
     - 🔴 Red: `Frustrated`
     - 🟣 Purple: `Bored`
     - 🔵 Cyan: `Drowsy`
     - ⚪ Gray: `Looking Away`
   - Encodes frame as JPEG and streams as multipart MJPEG HTTP stream.
3. REST Endpoints:
   - `GET /`: Serves the dashboard HTML.
   - `GET /video_feed`: Real-time MJPEG live camera stream.
   - `POST /api/session/start`: Starts session with optional timer.
   - `POST /api/session/stop`: Stops session and returns full pedagogical summary.
   - `GET /api/stats`: Real-time session data polled by the frontend.

---

### FILE 7: `frontend/index.html`
**Purpose:** Responsive Teacher Dashboard Interface.

#### Key Features:
- Clean, modern dashboard with dark theme and glassmorphism.
- Live video player showing bounding boxes and predictions directly on student faces.
- Real-time gauge for **Class Engagement Score (%)**.
- Dynamic bar charts showing current frame vs. overall session distribution.
- Session control toolbar: Start Session, Session Timer, Stop Session.
- Post-session modal dialog with classroom health score and AI teaching recommendations.

---

### FILE 8: `requirements.txt`
**Purpose:** Clean, Lightweight Dependency Specification.

```txt
scikit-learn     # Classical ML classifiers (SVM, RF, KNN), metrics, pipelines
opencv-python    # Haar Cascade face detection, image I/O, video capture, drawing
matplotlib       # Visualization of ROC curves, confusion matrix, learning curves
numpy            # High-performance numerical feature matrices
pillow           # Image processing primitives
pillow-heif      # Support for HEIC mobile photo conversion
flask            # Micro-web framework for dashboard and MJPEG stream
joblib           # High-performance persistence of scikit-learn models
scikit-image     # HOG and Local Binary Pattern (LBP) feature extractors
```
*(Notice: `tensorflow` has been completely eliminated.)*

---

## 4. Mastery of the 10 Core Machine Learning Concepts

When explaining your project to teachers or examiners, highlight these 10 concepts:

| # | Concept | Where in Code | Technical Explanation |
| :---: | :--- | :--- | :--- |
| **1** | **Supervised Learning** | `train.py` | Model learns from ground truth pairs $(X, y)$ where $X \in \mathbb{R}^{1930 \times 938}$ and $y \in \{0, 1, 2, 3, 4, 5\}$. |
| **2** | **Feature Engineering** | `model.py` | Replacing deep feature learning with manual domain-specific features: HOG (800) + LBP (90) + Color (32) + Haar (16) = 938 dims. |
| **3** | **Feature Normalization** | `train.py` / `inference.py` | `StandardScaler` calculates $z = \frac{x - \mu}{\sigma}$. Prevents features with large scale from dominating Euclidean distance in KNN and RBF kernel in SVM. |
| **4** | **Support Vector Machine (SVM)** | `train.py` (`SVC`) | Finds the optimal hyperplane with maximum geometric margin separating classes. Uses RBF kernel $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ to project into infinite-dimensional Hilbert space. |
| **5** | **Random Forest (Ensemble)** | `train.py`, `evaluate.py` | Ensemble of 200 decision trees built using Bagging (Bootstrap Aggregation) and feature sub-sampling. Produces Mean Decrease in Impurity (MDI) feature importances. |
| **6** | **K-Nearest Neighbors (KNN)** | `train.py` (`KNeighborsClassifier`) | Non-parametric lazy learner. Computes Euclidean distance to stored training points; predicts via inverse-distance-weighted majority voting ($k=7$). |
| **7** | **Cross-Validation** | `train.py` (`StratifiedKFold`) | 5-Fold cross-validation where each fold preserves exact class proportions to validate model stability and tune hyperparameters. |
| **8** | **Overfitting vs Underfitting** | `evaluate.py` (`learning_curve`) | Diagnosed via Learning Curves. A tight gap between training accuracy and cross-validation accuracy indicates strong generalization without overfitting. |
| **9** | **Evaluation Metrics** | `evaluate.py` | Precision ($\frac{TP}{TP+FP}$), Recall ($\frac{TP}{TP+FN}$), F1-Score ($2 \frac{P \cdot R}{P+R}$), Confusion Matrix, and Multi-Class One-vs-Rest ROC AUC. |
| **10** | **Hyperparameter Tuning** | `train.py` | Systematic parameter comparison ($C \in [0.1, 1, 10, 50]$, $n\_est \in [50, 100, 200]$, $k \in [3, 5, 7, 9]$) logged into comparison tables. |

---

## 5. Model Benchmark & Experimental Results

### Held-Out Test Set (347 Unseen Images)

| Classifier | Validation Accuracy | Test Accuracy | Weighted F1-Score | Macro ROC AUC | Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SVM (RBF, $C=10$)** 🏆 | **97.94%** | **93.08%** | **0.9306** | **0.995** | **2.70s** |
| **Random Forest ($n=200$)** | 98.23% | 91.35% | 0.9134 | 0.992 | 1.55s |
| **KNN ($k=7$, Distance-Weighted)** | 98.23% | 90.78% | 0.9069 | 0.989 | 0.03s |

### Per-Class Performance (SVM Champion Model)

```
              precision    recall  f1-score   support
       Bored      0.909     0.833     0.870        60
    Confused      0.966     0.918     0.941        61
      Drowsy      0.857     0.900     0.878        40
     Focused      0.892     0.951     0.921        61
  Frustrated      1.000     1.000     1.000        55
Looking Away      0.944     0.971     0.958        70

    accuracy                          0.931       347
   macro avg      0.928     0.929     0.928       347
weighted avg      0.932     0.931     0.931       347
```

---

## 6. Teacher Viva / Defense Q&A (Urdu + English)

Here are the exact questions your teachers will ask and how to answer them:

---

#### ❓ Q1: "Aapke project mein Machine Learning kahan hai? Kya yeh Deep Learning nahi hai?"
- **English Answer:**  
  "Sir/Ma'am, this is a purely Classical Machine Learning project. We do not use any Convolutional Neural Networks, Deep Learning backbones, backpropagation, or GPU training. Instead, we use classical **Feature Engineering**: we extract 938 hand-crafted visual features using **HOG (shape/gradients)**, **Spatial LBP (skin texture)**, **HSV Color Histograms (lighting-invariant color)**, and **Haar-like regional contrasts**. We then train classical ML classifiers—**Support Vector Machines (SVM)**, **Random Forest**, and **K-Nearest Neighbors (KNN)** using `scikit-learn`."
- **Roman Urdu Summary:**  
  *"Ma'am/Sir, isme koi deep learning ya neural network nahi hai. Humne CNN ki jagah classical feature engineering ki hai—HOG se edges, LBP se texture, HSV se color, aur Haar-like differences se face geometry extract karke 938 dimensions ka feature vector banaya hai. Phir scikit-learn ke classical models (SVM, Random Forest, KNN) train kiye hain."*

---

#### ❓ Q2: "Feature vector 938 dimensions ka kaise bana? Breakdown batao."
- **English Answer:**  
  "Every face crop is standardized to $96 \times 96$ pixels:
  1. **HOG (800 dims):** 8 gradient orientation bins across $(5 \times 5)$ blocks of $(2 \times 2)$ cells $\to 5 \times 5 \times 4 \times 8 = 800$.
  2. **Spatial LBP (90 dims):** $3 \times 3$ grid on the face, each computing a 10-bin uniform LBP histogram $\to 9 \times 10 = 90$.
  3. **HSV Color (32 dims):** 16 bins for Hue + 8 bins for Saturation + 8 bins for Value $\to 32$.
  4. **Haar-like Differences (16 dims):** Relative intensity differences (forehead vs. eyes, mouth vs. nose, left vs. right symmetry, 4-quadrant contrasts) $\to 16$.
  Total = $800 + 90 + 32 + 16 = \mathbf{938\text{ features}}$."
- **Roman Urdu Summary:**  
  *"Har face crop 96x96 ka hota hai. HOG ke 800 features facial contours dete hain, LBP ke 90 features skin texture dete hain, HSV ke 32 features lighting-invariant color dete hain, aur 16 Haar features aankh, naak aur maathay ke contrast differences dete hain. Sab concatenate hoke 938 dimensions banti hain."*

---

#### ❓ Q3: "`StandardScaler` lagana kyun zaroori tha? Agar na lagate toh kya hota?"
- **English Answer:**  
  "`StandardScaler` normalizes each feature to have mean $\mu = 0$ and variance $\sigma^2 = 1$. Both SVM (using the RBF kernel) and KNN rely on Euclidean distance $\|x_i - x_j\|^2$. If we don't scale features, a feature with high numerical variance (like Haar pixel sums) would completely dominate features with small numerical ranges (like normalized LBP probabilities), degrading accuracy."
- **Roman Urdu Summary:**  
  *"Agar StandardScaler na lagate toh jis feature ki numerical value badi hoti woh distance calculation pe haavi ho jati. SVM aur KNN Euclidean distance par depend karte hain, isliye sabhi 938 features ko zero mean aur unit variance pe lana mandatory tha."*

---

#### ❓ Q4: "Aapne SVM, Random Forest aur KNN kyun choose kiye? Inme se best kaun sa raha?"
- **English Answer:**  
  "We selected three foundational machine learning paradigms:
  1. **SVM (Margin-based):** Maximizes the margin between classes using the RBF kernel.
  2. **Random Forest (Ensemble/Bagging):** Combines 200 decorrelated decision trees, reducing variance and providing feature importance.
  3. **KNN (Instance-based / Lazy Learner):** Non-parametric distance-based voting.
  **SVM achieved the best performance** with **93.08% Test Accuracy** and a **0.995 ROC AUC**, because the RBF kernel projects the 938-dim feature space into a higher-dimensional space where facial engagement states are cleanly separable."
- **Roman Urdu Summary:**  
  *"Humne 3 alag machine learning families compare ki: SVM (margin-based), Random Forest (ensemble/tree-based), aur KNN (distance-based). Sabse best **SVM** raha jisne test set par **93.08% accuracy** aur **0.995 ROC AUC** achieve kiya."*

---

#### ❓ Q5: "Overfitting aur Underfitting kaise check kiya?"
- **English Answer:**  
  "We evaluated Learning Curves in `src/evaluate.py`. We plotted Training Accuracy vs. 5-Fold Cross-Validation Accuracy across sample sizes from 20% to 100%. The training score converged around 98% while cross-validation converged at 93.5% with a narrow generalization gap (~4.5%), proving our model does not suffer from high variance (overfitting) nor high bias (underfitting)."
- **Roman Urdu Summary:**  
  *"Humne `evaluate.py` mein Learning Curves plot kiye. Training curve aur Cross-Validation curve ke darmiyan sirf 4.5% ka gap hai. Agar training 100% hoti aur validation 60% hoti toh overfitting hoti; agar dono 60% hoti toh underfitting hoti. Hamara model perfectly regularized hai."*

---

#### ❓ Q6: "Cross-validation mein K-Fold ki jagah 'StratifiedKFold' kyun use kiya?"
- **English Answer:**  
  "Standard K-Fold randomly splits data, which can accidentally place fewer minority class samples in a given fold. **StratifiedKFold** guarantees that every single fold preserves the exact class distribution percentages of the overall dataset, ensuring unbiased validation."
- **Roman Urdu Summary:**  
  *"StratifiedKFold har fold mein class ka ratio barabar rakhta hai taake kisi fold mein koi class kam ya zyada na ho jaye."*

---

#### ❓ Q7: "Face Detection ke liye kya use kiya hai?"
- **English Answer:**  
  "We use OpenCV's classical **Haar Cascade Classifier** (`haarcascade_frontalface_default.xml`), based on the seminal Viola-Jones algorithm. It uses integral images and AdaBoost cascaded classifiers to detect faces in real-time without deep learning."
- **Roman Urdu Summary:**  
  *"Face detection ke liye hum OpenCV ka classical Haar Cascade (Viola-Jones algorithm) use kar rahe hain jo purely classical ML approach hai."*

---

#### ❓ Q8: "System ko run kaise karte hain?"
- **Answer:**  
  1. Train models: `python src/train_ml.py`
  2. Generate diagnostic graphs: `python src/evaluate.py`
  3. Start live dashboard: `python src/app.py` $\to$ opens `http://localhost:5000`
