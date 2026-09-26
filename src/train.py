"""
train.py
--------
Trains the engagement CNN on the prepared dataset (data/prepared/{train,val,test}).

Run:
    python src/train.py --data_dir data/prepared --epochs 30
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from model import build_model, IMG_SIZE


def make_dataset(directory: Path, batch_size: int, shuffle: bool):
    ds = keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=shuffle,
        seed=42,
    )
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/prepared")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--resume", action="store_true",
                     help="Continue training from models/final_model.keras + outputs/history.json if present")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset(data_dir / "train", args.batch_size, shuffle=True)
    val_ds = make_dataset(data_dir / "val", args.batch_size, shuffle=False)

    class_names = train_ds.class_names
    print("Classes:", class_names)

    # Save class order — the API/frontend must use this exact order later
    with open(model_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    history_path = out_dir / "history.json"
    prior_history = {}
    resume_path = model_dir / "final_model.keras"
    if args.resume and resume_path.exists():
        print(f"Resuming from {resume_path}")
        model = keras.models.load_model(resume_path)
        if history_path.exists():
            with open(history_path) as f:
                prior_history = json.load(f)
    else:
        model = build_model(num_classes=len(class_names))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            str(model_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Save final model too (best_model.keras already holds the best checkpoint)
    model.save(model_dir / "final_model.keras")

    # ---- Merge with any prior history (for --resume chunks) & save ----
    hist = history.history
    if prior_history:
        for k in hist:
            hist[k] = prior_history.get(k, []) + hist[k]
    with open(history_path, "w") as f:
        json.dump(hist, f)

    # ---- Plots: accuracy & loss curves (for spotting under/overfitting) ----
    epochs_range = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, hist["accuracy"], label="Train Accuracy")
    plt.plot(epochs_range, hist["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, hist["loss"], label="Train Loss")
    plt.plot(epochs_range, hist["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    print(f"\nSaved training curves to {out_dir / 'training_curves.png'}")
    print(f"Best model saved to {model_dir / 'best_model.keras'}")


if __name__ == "__main__":
    main()
