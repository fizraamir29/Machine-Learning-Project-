"""
model.py
--------
CNN for 6-class student engagement classification, built via TRANSFER
LEARNING on a frozen MobileNetV2 backbone (pretrained on ImageNet).

Why transfer learning instead of a from-scratch CNN:
Our merged dataset has ~1,500 training images across 6 classes (~250/class).
A from-scratch CNN with BatchNorm needs many more steps to build stable
running statistics — with this little data, BatchNorm's running mean/
variance collapse, and the model ends up predicting a single class for
every input at inference time regardless of what it's shown (verified:
val accuracy stuck exactly at the majority-class proportion, softmax
outputs nearly identical across different images).

Freezing a pretrained backbone sidesteps this entirely: MobileNetV2's
BatchNorm statistics come from ImageNet (millions of images) and are
never updated, so they're stable from step one. Only a small classifier
head is trained on our data, which is both more robust and much less
prone to overfitting with this dataset size.
"""
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (128, 128)
NUM_CLASSES = 6

# Local copy of the MobileNetV2 ImageNet weights (no top / no classifier head).
# Bundled in this project under pretrained/ so training works fully OFFLINE
# after the first setup -- no need to hit storage.googleapis.com (which is
# blocked on some networks/sandboxes) every time you train.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_WEIGHTS_PATH = os.path.join(_THIS_DIR, "..", "pretrained", "mobilenet_v2_no_top.h5")


def build_model(num_classes: int = NUM_CLASSES, img_size=IMG_SIZE,
                 fine_tune: bool = False, fine_tune_at: int = 100) -> keras.Model:
    inputs = keras.Input(shape=(*img_size, 3))

    # Light on-the-fly augmentation (only active during training)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomBrightness(0.1)(x)

    # MobileNetV2 expects inputs preprocessed to [-1, 1], NOT /255
    x = preprocess_input(x)

    if os.path.exists(LOCAL_WEIGHTS_PATH):
        # Build with no weights first, then load the bundled local .h5 file.
        # Avoids a runtime download from storage.googleapis.com entirely.
        base_model = keras.applications.MobileNetV2(
            input_shape=(*img_size, 3),
            include_top=False,
            weights=None,
        )
        base_model.load_weights(LOCAL_WEIGHTS_PATH)
    else:
        # Fallback: normal Keras download (needs internet access to
        # storage.googleapis.com). Only used if the local file is missing.
        base_model = keras.applications.MobileNetV2(
            input_shape=(*img_size, 3),
            include_top=False,
            weights="imagenet",
        )
    base_model.trainable = fine_tune
    if fine_tune:
        # Keep the early, more general layers frozen; only unfreeze the
        # later, more task-specific layers for optional fine-tuning.
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    # training=False keeps BatchNorm layers in inference mode even if the
    # backbone is later set trainable=True for fine-tuning — this is the
    # key line that prevents the collapse described above.
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="engagement_mobilenetv2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3 if not fine_tune else 1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
    print(f"\nTrainable params: {sum(p.numpy().size for p in m.trainable_weights):,}")
