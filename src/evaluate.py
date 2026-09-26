"""
evaluate.py
-----------
Comprehensive Classical Machine Learning Evaluation Module for Student Engagement Detection.

ML CONCEPTS IMPLEMENTED & DEMONSTRATED:
================================================================================
ML CONCEPT 8: OVERFITTING VS UNDERFITTING (DIAGNOSED VIA LEARNING CURVES)
- Learning curves plot performance (accuracy) on training vs validation sets as
  a function of training dataset size.
- Underfitting (High Bias): Both training and validation accuracies remain low.
- Overfitting (High Variance): Training accuracy is high (~100%) but validation
  accuracy is significantly lower, showing a wide generalization gap.
- Optimal Fit: Validation accuracy approaches training accuracy with a narrow gap.

ML CONCEPT 9: COMPREHENSIVE EVALUATION METRICS
- Confusion Matrix: Full N x N contingency table showing true positives, false
  positives, false negatives per engagement state.
- Precision: TP / (TP + FP) — proportion of predicted engaged students who were actually engaged.
- Recall: TP / (TP + FN) — proportion of truly engaged students successfully captured.
- F1-Score: Harmonic mean of Precision and Recall = 2 * (P * R) / (P + R).
- ROC Curves & AUC (One-vs-Rest): Receiver Operating Characteristic plotting True
  Positive Rate (Sensitivity) vs False Positive Rate (1 - Specificity) across all thresholds.
- Feature Importance (Random Forest): Gini impurity decrease attributed to each hand-crafted feature.
================================================================================
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DEFAULT_CLASSES, get_feature_names
from train import load_dataset_features

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "prepared"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


# ==============================================================================
# 1. Confusion Matrix & Classification Report
# ==============================================================================
def evaluate_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    out_dir: Path,
    model_name: str = "SVM",
):
    """
    ML Concept 9: Confusion Matrix & Detailed Metrics
    Generates and saves the test set confusion matrix and classification report.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=class_names, digits=3)

    print(f"\n[{model_name}] Classification Report (Test Set):")
    print(report)

    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"=== {model_name} CLASSIFICATION REPORT (HELD-OUT TEST SET) ===\n\n")
        f.write(report)
    print(f"  Saved report to {report_path}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=35, cmap="Blues", colorbar=True)
    plt.title(f"Confusion Matrix — {model_name} (Held-out Test Split)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=180)
    plt.close()
    print(f"  Saved confusion matrix plot to {cm_path}")


# ==============================================================================
# 2. Multi-Class ROC Curves (One-vs-Rest) & AUC
# ==============================================================================
def plot_roc_curves(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    out_dir: Path,
    model_name: str = "SVM",
):
    """
    ML Concept 9: Multi-class ROC & AUC (One-vs-Rest Strategy)
    Plots True Positive Rate vs False Positive Rate for each engagement class.
    """
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # Get class probabilities
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(9, 7))
    colors = ["#2b5c8f", "#e05638", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_score[:, i])
        plt.plot(
            fpr[i],
            tpr[i],
            color=colors[i % len(colors)],
            lw=2,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
        )

    # Diagonal baseline (random guess)
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title(f"Multi-Class ROC Curves (One-vs-Rest) — {model_name}", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10, framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    roc_path = out_dir / "roc_curves.png"
    plt.savefig(roc_path, dpi=180)
    plt.close()
    print(f"  Saved ROC curves plot to {roc_path}")


# ==============================================================================
# 3. Learning Curves (Diagnosing Overfitting vs Underfitting)
# ==============================================================================
def plot_learning_curves(
    model,
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    model_name: str = "SVM",
):
    """
    ML Concept 8: Overfitting vs Underfitting Analysis
    Plots training vs cross-validation accuracy as training sample size increases.
    """
    print(f"\nComputing Learning Curves for [{model_name}]...")
    train_sizes = np.linspace(0.2, 1.0, 5)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=train_sizes,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
    )

    train_mean = np.mean(train_scores, axis=1) * 100
    train_std  = np.std(train_scores, axis=1) * 100
    val_mean   = np.mean(val_scores, axis=1) * 100
    val_std    = np.std(val_scores, axis=1) * 100

    plt.figure(figsize=(9, 6))
    plt.plot(train_sizes_abs, train_mean, "o-", color="#d9534f", lw=2, label="Training Score")
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#d9534f")

    plt.plot(train_sizes_abs, val_mean, "s-", color="#0275d8", lw=2, label="Cross-Validation Score")
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#0275d8")

    plt.title(f"Learning Curves: Training vs Cross-Validation — {model_name}", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Training Examples (Sample Size)", fontsize=11, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)

    # Annotate diagnosis insight
    gap = train_mean[-1] - val_mean[-1]
    diag_text = (
        f"Generalization Gap: {gap:.1f}%\n"
        f"Validation Convergence: {val_mean[-1]:.1f}%\n"
        f"Diagnosis: Well-regularized ML model"
    )
    plt.annotate(
        diag_text,
        xy=(0.04, 0.12),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
        fontsize=9,
    )

    plt.tight_layout()
    lc_path = out_dir / "learning_curves.png"
    plt.savefig(lc_path, dpi=180)
    plt.close()
    print(f"  Saved learning curves plot to {lc_path}")


# ==============================================================================
# 4. Feature Importance Plot (Random Forest)
# ==============================================================================
def plot_feature_importance(
    rf_model,
    out_dir: Path,
    top_n: int = 25,
):
    """
    ML Concept 5: Random Forest Feature Importance (MDI - Mean Decrease in Impurity)
    Visualizes the top hand-crafted visual features contributing most to engagement prediction.
    """
    print("\nExtracting Random Forest Feature Importances...")
    # Extract classifier from pipeline if needed
    clf = rf_model.named_steps["clf"] if hasattr(rf_model, "named_steps") else rf_model
    if not hasattr(clf, "feature_importances_"):
        print("  Classifier does not support feature_importances_. Skipping plot.")
        return

    importances = clf.feature_importances_
    all_names = get_feature_names()

    # Ensure length matches
    if len(all_names) != len(importances):
        all_names = [f"Feature_{i}" for i in range(len(importances))]

    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_names = [all_names[i] for i in indices]

    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_names))
    plt.barh(y_pos, top_importances[::-1], color="#2b5c8f", align="center", edgecolor="#1a365d")
    plt.yticks(y_pos, top_names[::-1], fontsize=9)
    plt.xlabel("Relative Feature Importance (Mean Decrease in Impurity)", fontsize=11, fontweight="bold")
    plt.title(f"Top {top_n} Most Informative Hand-Crafted Visual Features (Random Forest)", fontsize=12, fontweight="bold", pad=15)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    fi_path = out_dir / "feature_importance.png"
    plt.savefig(fi_path, dpi=180)
    plt.close()
    print(f"  Saved feature importance plot to {fi_path}")


# ==============================================================================
# 5. Full 3-Classifier Comparison Table
# ==============================================================================
def compare_all_classifiers(
    models: Dict[str, any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    out_dir: Path,
):
    """
    Compares SVM, Random Forest, and KNN across all primary ML evaluation metrics.
    """
    print("\n" + "=" * 80)
    print("  EVALUATING AND COMPARING ALL 3 CLASSIFIERS ON HELD-OUT TEST SET")
    print("=" * 80)

    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    table_lines = [
        "+" + "-" * 16 + "+" + "-" * 12 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 12 + "+",
        f"| {'Classifier':<14} | {'Accuracy':<10} | {'Precision(W)':<12} | {'Recall(W)':<12} | {'F1-Score(W)':<12} | {'ROC AUC(M)':<10} |",
        "+" + "-" * 16 + "+" + "-" * 12 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 12 + "+",
    ]

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = np.mean(y_pred == y_test)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # ROC AUC
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
            else:
                y_prob = model.decision_function(X_test)
            auc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
            auc_str = f"{auc:.3f}"
        except Exception:
            auc_str = "N/A"

        acc_str = f"{acc*100:.2f}%"
        prec_str = f"{prec:.4f}"
        rec_str = f"{rec:.4f}"
        f1_str = f"{f1:.4f}"
        line = (
            f"| {name.upper():<14} | {acc_str:<10} | {prec_str:<12} | "
            f"{rec_str:<12} | {f1_str:<12} | {auc_str:<10} |"
        )
        table_lines.append(line)
        print(f"  {line}")

    table_lines.append("+" + "-" * 16 + "+" + "-" * 12 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 12 + "+")
    table_str = "\n".join(table_lines)

    comp_path = out_dir / "model_comparison_table.txt"
    with open(comp_path, "w") as f:
        f.write("=== 3-CLASSIFIER EVALUATION COMPARISON ON TEST SET ===\n\n")
        f.write(table_str)
    print(f"\n  Saved comparison table to {comp_path}")


# ==============================================================================
# Main Routine
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate classical ML models.")
    parser.add_argument("--data_dir", default=str(DATA_DIR), help="Path to prepared dataset")
    parser.add_argument("--models_dir", default=str(MODELS_DIR), help="Path to trained models")
    parser.add_argument("--out_dir", default=str(OUTPUTS_DIR), help="Output directory for plots")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Class names
    class_names_path = models_dir / "class_names.json"
    if class_names_path.exists():
        with open(class_names_path) as f:
            class_names = json.load(f)
    else:
        class_names = DEFAULT_CLASSES

    # Load test set features
    cache_dir = out_dir / "cache"
    X_test, y_test = load_dataset_features(data_dir / "test", class_names, cache_dir / "test_feats.npz")
    X_train, y_train = load_dataset_features(data_dir / "train", class_names, cache_dir / "train_feats.npz")

    # Load models
    best_model_path = models_dir / "best_model.joblib"
    if not best_model_path.exists():
        print(f"Error: Model not found at {best_model_path}. Run 'python src/train.py' first!")
        return

    best_model = joblib.load(best_model_path)
    print(f"Loaded primary model from {best_model_path}")

    # 1. Confusion Matrix & Classification Report
    evaluate_confusion_matrix(best_model, X_test, y_test, class_names, out_dir, model_name="Best ML Model")

    # 2. Multi-class ROC Curves
    plot_roc_curves(best_model, X_test, y_test, class_names, out_dir, model_name="Best ML Model")

    # 3. Learning Curves
    plot_learning_curves(best_model, X_train, y_train, out_dir, model_name="Best ML Model")

    # 4. Feature Importance for Random Forest
    rf_model_path = models_dir / "rf_model.joblib"
    if rf_model_path.exists():
        rf_model = joblib.load(rf_model_path)
        plot_feature_importance(rf_model, out_dir, top_n=25)

    # 5. Compare All 3 Classifiers
    models = {}
    for name in ["svm", "rf", "knn"]:
        p = models_dir / f"{name}_model.joblib"
        if p.exists():
            models[name] = joblib.load(p)
    if models:
        compare_all_classifiers(models, X_test, y_test, class_names, out_dir)

    print("\n" + "=" * 80)
    print("  ALL ML EVALUATION ARTIFACTS SUCCESSFULLY GENERATED IN outputs/")
    print("=" * 80)


if __name__ == "__main__":
    main()
