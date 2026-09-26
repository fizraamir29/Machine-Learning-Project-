"""
evaluate.py
-----------
Loads the trained model and evaluates it on the held-out test split.
Produces: accuracy/loss on test set, confusion matrix image, and a
classification report (precision/recall/F1 per class).

Run:
    python src/evaluate.py --data_dir data/prepared --model_path models/best_model.keras
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

from model import IMG_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/prepared")
    ap.add_argument("--model_path", default="models/best_model.keras")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(args.model_path).parent / "class_names.json") as f:
        class_names = json.load(f)

    test_ds = keras.utils.image_dataset_from_directory(
        data_dir / "test",
        image_size=IMG_SIZE,
        batch_size=32,
        label_mode="categorical",
        shuffle=False,
    )
    # image_dataset_from_directory infers its own class order from the folder
    # names again here — assert it matches what training used, else the
    # confusion matrix labels would be silently wrong.
    assert test_ds.class_names == class_names, (
        f"Class order mismatch: test={test_ds.class_names} vs train={class_names}"
    )

    model = keras.models.load_model(args.model_path)

    test_ds_eval = test_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_loss, test_acc = model.evaluate(test_ds_eval)
    print(f"\nTest accuracy: {test_acc:.4f}   Test loss: {test_loss:.4f}")

    y_true = []
    y_pred = []
    for batch_images, batch_labels in test_ds:
        preds = model.predict(batch_images, verbose=0)
        y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print("\nClassification report:\n", report)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\nTest loss: {test_loss:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    print(f"\nSaved confusion matrix to {out_dir / 'confusion_matrix.png'}")
    print(f"Saved classification report to {out_dir / 'classification_report.txt'}")


if __name__ == "__main__":
    main()
