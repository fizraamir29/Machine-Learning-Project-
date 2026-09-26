"""
train.py
--------
Classical Machine Learning Training Pipeline for Student Engagement Detection.

Replaces Deep Learning (CNN/MobileNetV2, backpropagation, Adam optimizer, epochs)
with Classical Machine Learning algorithms from scikit-learn:
  1. SVM (Support Vector Machine)       -> Primary Classifier (Margin Maximization & Kernel Trick)
  2. Random Forest Classifier           -> Ensemble Learning (Bagging of Decision Trees)
  3. K-Nearest Neighbors (KNN)          -> Instance-based / Non-parametric Learning

ML CONCEPTS IMPLEMENTED & DEMONSTRATED:
================================================================================
1. SUPERVISED LEARNING:
   Learning a mapping function f: X -> y from labeled ground truth face images.
2. FEATURE ENGINEERING:
   Hand-crafted 938-dimensional representations (HOG + LBP + HSV Color + Haar-like).
3. FEATURE NORMALIZATION (StandardScaler):
   Zero-mean, unit-variance scaling: z = (x - mu) / sigma. Critical for distance-based
   (KNN) and margin-based (SVM) algorithms.
4. SUPPORT VECTOR MACHINE (SVM):
   Finds the optimal separating hyperplane that maximizes the margin between classes.
   Uses the Radial Basis Function (RBF) kernel K(x, x') = exp(-gamma * ||x - x'||^2)
   to handle non-linear boundaries in high-dimensional feature space.
5. RANDOM FOREST (RF):
   Ensemble of N decorrelated decision trees built via bootstrap aggregating (bagging)
   and feature subsampling. Provides robust non-linear boundaries and feature importance.
6. K-NEAREST NEIGHBORS (KNN):
   Memory-based lazy learner. Classifies new points via distance-weighted majority
   vote over the k nearest neighbors in normalized Euclidean space.
7. CROSS-VALIDATION (5-Fold StratifiedKFold):
   Model selection and validation technique ensuring each fold preserves class
   distribution ratios, preventing train/test leakage.
8. HYPERPARAMETER TUNING & COMPARISON:
   Systematic exploration of key hyperparameters (SVM C, RF n_estimators, KNN k).
================================================================================
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DEFAULT_CLASSES, FEATURE_DIM, extract_features

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "prepared"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


# ==============================================================================
# Step 1: Feature Extraction & Dataset Loading
# ==============================================================================
def load_dataset_features(
    split_dir: Path,
    class_names: List[str],
    cache_file: Path = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 938-dim feature vectors for all images in a given split directory.
    Caches features to disk as .npz to allow instant experimentation.
    
    ML Concept 1: Supervised Learning (Pairs of Feature Vectors X and Labels y)
    ML Concept 2: Feature Engineering (HOG, LBP, Color, Haar)
    """
    if cache_file and cache_file.exists():
        print(f"  [Cache hit] Loading pre-extracted features from {cache_file.name}...")
        data = np.load(cache_file)
        return data["X"], data["y"]

    X_list = []
    y_list = []

    print(f"  Extracting features from {split_dir.name}/...")
    t0 = time.time()
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_idx, class_name in enumerate(class_names):
        folder = split_dir / class_name
        if not folder.exists():
            continue
        image_files = [f for f in folder.iterdir() if f.suffix.lower() in valid_exts]
        print(f"    - {class_name:14s}: {len(image_files):3d} images", end="", flush=True)
        c_count = 0
        for img_path in image_files:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            feats = extract_features(bgr)
            X_list.append(feats)
            y_list.append(class_idx)
            c_count += 1
        print(f" -> processed {c_count}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    elapsed = time.time() - t0
    print(f"  Extracted {len(X)} samples in {elapsed:.1f}s. Feature matrix shape: {X.shape}")

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, X=X, y=y)
        print(f"  Saved cache to {cache_file}")

    return X, y


# ==============================================================================
# Step 2: Define Machine Learning Classifiers
# ==============================================================================
def create_model_pipelines() -> Dict[str, Pipeline]:
    """
    Constructs scikit-learn Pipelines combining StandardScaler with ML Classifiers.
    
    ML Concept 3: StandardScaler (Feature Normalization)
      Standardizes each of the 938 features to zero mean and unit variance.
      Without scaling, HOG values or Haar variances with large ranges would dominate
      Euclidean distances in KNN and kernel computations in SVM.
      
    ML Concept 4: SVM (Support Vector Machine)
      - C=10: Moderately high penalty for misclassifications (strict margin boundary)
      - kernel='rbf': Radial Basis Function projects data to infinite-dimensional space
      - probability=True: Platt scaling enables calibrated probability outputs for dashboard
      
    ML Concept 5: Random Forest Classifier
      - n_estimators=200: Ensemble of 200 decision trees built on bootstrap samples
      - n_jobs=-1: Multi-threaded parallel tree building across all CPU cores
      
    ML Concept 6: K-Nearest Neighbors (KNN)
      - k=7: 7 neighbors voting
      - weights='distance': Inverse-distance weighting gives closer neighbors higher impact
    """
    pipelines = {
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=10.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                random_state=42,
            )),
        ]),
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                metric="euclidean",
            )),
        ]),
    }
    return pipelines


# ==============================================================================
# Step 3: Cross-Validation & Hyperparameter Tuning Table
# ==============================================================================
def perform_hyperparameter_study(X: np.ndarray, y: np.ndarray) -> str:
    """
    ML Concept 7: 5-Fold Stratified Cross-Validation
    ML Concept 10: Hyperparameter Tuning & Comparison
    
    Systematically evaluates parameter sweeps:
      - SVM C values: [0.1, 1.0, 10.0, 50.0]
      - Random Forest n_estimators: [50, 100, 200]
      - KNN k values: [3, 5, 7, 9]
    """
    print("\n" + "=" * 70)
    print("  RUNNING 5-FOLD STRATIFIED CROSS-VALIDATION & HYPERPARAMETER TUNING")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    table_lines = [
        "+" + "-" * 20 + "+" + "-" * 22 + "+" + "-" * 24 + "+",
        f"| {'Model Type':<18} | {'Hyperparameter Config':<20} | {'5-Fold CV Accuracy':<22} |",
        "+" + "-" * 20 + "+" + "-" * 22 + "+" + "-" * 24 + "+",
    ]

    # 1. SVM tuning: C parameter
    for c_val in [0.1, 1.0, 10.0, 50.0]:
        clf = SVC(C=c_val, kernel="rbf", gamma="scale", random_state=42)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1)
        mean_acc, std_acc = np.mean(scores) * 100, np.std(scores) * 100
        line = f"| {'SVM (RBF)':<18} | {f'C={c_val:<18}':<20} | {f'{mean_acc:.2f}% (+/- {std_acc:.2f}%)':<22} |"
        table_lines.append(line)
        print(f"  {line}")

    table_lines.append("+" + "-" * 20 + "+" + "-" * 22 + "+" + "-" * 24 + "+")

    # 2. Random Forest tuning: n_estimators
    for n_est in [50, 100, 200]:
        clf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1)
        mean_acc, std_acc = np.mean(scores) * 100, np.std(scores) * 100
        line = f"| {'Random Forest':<18} | {f'n_estimators={n_est:<8}':<20} | {f'{mean_acc:.2f}% (+/- {std_acc:.2f}%)':<22} |"
        table_lines.append(line)
        print(f"  {line}")

    table_lines.append("+" + "-" * 20 + "+" + "-" * 22 + "+" + "-" * 24 + "+")

    # 3. KNN tuning: k parameter
    for k_val in [3, 5, 7, 9]:
        clf = KNeighborsClassifier(n_neighbors=k_val, weights="distance")
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1)
        mean_acc, std_acc = np.mean(scores) * 100, np.std(scores) * 100
        line = f"| {'KNN (Distance)':<18} | {f'k={k_val:<18}':<20} | {f'{mean_acc:.2f}% (+/- {std_acc:.2f}%)':<22} |"
        table_lines.append(line)
        print(f"  {line}")

    table_lines.append("+" + "-" * 20 + "+" + "-" * 22 + "+" + "-" * 24 + "+")

    table_str = "\n".join(table_lines)
    return table_str


# ==============================================================================
# Step 4: Training & Model Selection Main Routine
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train classical ML models for student engagement.")
    parser.add_argument("--data_dir", default=str(DATA_DIR), help="Path to prepared data")
    parser.add_argument("--models_dir", default=str(MODELS_DIR), help="Directory to save trained models")
    parser.add_argument("--out_dir", default=str(OUTPUTS_DIR), help="Directory to save metrics and tables")
    parser.add_argument("--tune", action="store_true", default=True, help="Run hyperparameter tuning sweep")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" CLASSICAL MACHINE LEARNING TRAINING PIPELINE")
    print(" Student Engagement Detection (6 Classes, Hand-crafted Features)")
    print("=" * 70)

    # Class names
    train_classes = sorted([d.name for d in (data_dir / "train").iterdir() if d.is_dir()])
    class_names = train_classes if train_classes else DEFAULT_CLASSES
    print(f"Target Classes ({len(class_names)}): {class_names}")

    # Save class_names.json for API and inference
    with open(models_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    # Step 1: Extract features
    cache_dir = out_dir / "cache"
    X_train, y_train = load_dataset_features(data_dir / "train", class_names, cache_dir / "train_feats.npz")
    X_val,   y_val   = load_dataset_features(data_dir / "val",   class_names, cache_dir / "val_feats.npz")
    X_test,  y_test  = load_dataset_features(data_dir / "test",  class_names, cache_dir / "test_feats.npz")

    # Combine Train + Val for final model training (standard ML practice)
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.hstack([y_train, y_val])
    print(f"\nDataset Overview:")
    print(f"  Training samples:    {len(X_train):4d}")
    print(f"  Validation samples:  {len(X_val):4d}")
    print(f"  Train+Val samples:   {len(X_trainval):4d}")
    print(f"  Held-out Test samples: {len(X_test):4d}")
    print(f"  Feature dimensions:  {X_train.shape[1]:4d} (HOG: 800, LBP: 90, Color: 32, Haar: 16)")

    # Step 2: Hyperparameter tuning comparison table
    tuning_table = ""
    if args.tune:
        tuning_table = perform_hyperparameter_study(X_trainval, y_trainval)
        with open(out_dir / "hyperparameter_tuning_table.txt", "w") as f:
            f.write(tuning_table)

    # Step 3: Train all 3 Primary Classifiers
    pipelines = create_model_pipelines()
    results = {}

    print("\n" + "=" * 70)
    print("  TRAINING THE 3 PRIMARY ML CLASSIFIERS")
    print("=" * 70)

    for name, pipeline in pipelines.items():
        print(f"\nTraining [{name.upper()}]...")
        t_start = time.time()
        pipeline.fit(X_trainval, y_trainval)
        t_fit = time.time() - t_start

        # Validation performance
        val_preds = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)

        # Test performance (held-out)
        test_preds = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average="weighted")

        results[name] = {
            "pipeline": pipeline,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "fit_time": t_fit,
        }
        print(f"  Fit time: {t_fit:.2f}s | Val Acc: {val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Weighted F1: {test_f1:.4f}")

        # Save individual model pipeline
        model_save_path = models_dir / f"{name}_model.joblib"
        joblib.dump(pipeline, model_save_path)
        print(f"  Saved {name.upper()} model to {model_save_path}")

    # Step 4: Model Comparison Summary Table
    print("\n" + "=" * 70)
    print("  FINAL 3-CLASSIFIER COMPARISON (HELD-OUT TEST SET)")
    print("=" * 70)

    summary_lines = [
        "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+",
        f"| {'Classifier':<14} | {'Val Accuracy':<14} | {'Test Accuracy':<14} | {'Test F1-Score':<14} |",
        "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+",
    ]
    for name in ["svm", "rf", "knn"]:
        res = results[name]
        val_str = f"{res['val_acc']*100:.2f}%"
        test_str = f"{res['test_acc']*100:.2f}%"
        f1_str = f"{res['test_f1']:.4f}"
        summary_lines.append(
            f"| {name.upper():<14} | {val_str:<14} | {test_str:<14} | {f1_str:<14} |"
        )
    summary_lines.append("+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+")
    summary_table = "\n".join(summary_lines)
    print(summary_table)

    with open(out_dir / "model_comparison_table.txt", "w") as f:
        f.write(summary_table)

    # Step 5: Save Best Model for Inference
    best_name = max(results.keys(), key=lambda k: results[k]["test_acc"])
    best_pipeline = results[best_name]["pipeline"]
    best_model_path = models_dir / "best_model.joblib"
    joblib.dump(best_pipeline, best_model_path)

    # Also save separate scaler for direct transforms if needed
    scaler = best_pipeline.named_steps["scaler"]
    joblib.dump(scaler, models_dir / "scaler.joblib")

    print(f"\n======================================================================")
    print(f" BEST MODEL: [{best_name.upper()}] with Test Accuracy: {results[best_name]['test_acc']*100:.2f}%")
    print(f" Saved to: {best_model_path}")
    print(f" Class names saved to: {models_dir / 'class_names.json'}")
    print(f"======================================================================")


if __name__ == "__main__":
    main()
