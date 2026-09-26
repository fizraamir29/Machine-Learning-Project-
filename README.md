# Interactive Classroom Engagement Detection — Complete System

Real-time student engagement detection from a webcam feed, with a live
teacher dashboard. **The model is already trained and included** —
just install requirements and run `src/app.py`.

## What's included (already done)

- **Merged dataset** from 3 sources into one unified 6-class set:
  - Kaggle "Student-engagement-dataset" (2,120 images)
  - Custom "Maleha" dataset (32 images)
  - Custom "Zoha" dataset (125 images, HEIC auto-converted to JPG)
  - **Total: 2,277 real images**, split 70/15/15 → `data/prepared/{train,val,test}`
- **A trained model** at `models/best_model.keras` (transfer learning on
  a frozen, pretrained MobileNetV2 backbone) — no need to retrain unless
  you want to
- **Real evaluation results** on the held-out test set (347 images, see
  `outputs/`):
  - Test accuracy: **87.9%**
  - Per-class precision/recall/F1 → `outputs/classification_report.txt`
  - Confusion matrix → `outputs/confusion_matrix.png`
  - Train/val accuracy & loss curves → `outputs/training_curves.png`
- **Live camera + dashboard system**: face detection → per-face
  engagement prediction → boxes drawn live on video → session stats

Final classes: `Bored, Confused, Drowsy, Focused, Frustrated, Looking Away`

## 1. Setup

```bash
pip install -r requirements.txt
```

No internet needed to train — this project bundles the pretrained
MobileNetV2 ImageNet weights locally at `pretrained/mobilenet_v2_no_top.h5`
(fetched once from a GitHub-hosted mirror, since Keras's default source,
`storage.googleapis.com`, is blocked on some networks/sandboxes).
`src/model.py` loads this local file automatically — you'll only hit the
network fallback if that file is missing.

## 2. Run the live system (camera + dashboard) — the main thing

```bash
python src/app.py
```

This starts a local Flask server and opens `http://localhost:5000` in
your browser automatically — the teacher dashboard.

**How it works:**
- Click **Start Session** (optionally name it) — this opens your laptop
  webcam feed in the dashboard
- Every frame: OpenCV Haar cascade detects every face in view → each
  face is cropped → fed to the trained model → predicted engagement
  label is drawn as a colored box directly on the video (green = an
  "engaged" state, red = a "not engaged" state)
- The side panel shows a **live per-frame breakdown** and a **running
  session-total breakdown**, updating every second
- Click **Stop Session** — a summary (duration, per-class percentages,
  total data points) appears

**Notes:**
- Works with multiple faces in frame at once (real classroom use case)
  — each face gets its own box and label independently
- Camera permission: your OS/browser may prompt for webcam access the
  first time — allow it. If `cv2.VideoCapture(0)` fails to open, check
  no other app (Zoom, Teams, etc.) is currently using the camera
- The in-memory session log resets each time you restart the server
  (fine for a class demo)

## 3. (Optional) Retrain from scratch

Only needed if you add more images or want to tweak the model — the
shipped `models/best_model.keras` already works out of the box.

```bash
# 3a. Re-merge + re-split the raw datasets (only if you changed the raw folders)
python src/prepare_data.py \
    --kaggle_dir "/path/to/Student-engagement-dataset" \
    --maleha_dir "/path/to/Maleha/Crop_images" \
    --zoha_dir   "/path/to/Zoha" \
    --out_dir data/prepared

# 3b. Train (fresh run)
python src/train.py --data_dir data/prepared --epochs 15

#     ...or continue training an existing model for more epochs:
python src/train.py --data_dir data/prepared --epochs 8 --resume

# 3c. Evaluate on the test set
python src/evaluate.py --data_dir data/prepared --model_path models/best_model.keras
```

`train.py` trains only a small classifier head (~165K params) on top of
the **frozen** pretrained MobileNetV2 backbone. This is deliberate: an
earlier attempt trained a CNN with BatchNorm from scratch on this small
dataset, and it collapsed (validation accuracy stuck at the
majority-class rate, loss exploding) because BatchNorm needs far more
data/steps than ~1,500 images to build stable running statistics.
Freezing a backbone pretrained on millions of ImageNet images sidesteps
that entirely — only the head is trained, which is both more robust and
far less prone to overfitting at this dataset size. This is also the
standard, textbook-recommended approach for small image datasets, so
it's a legitimate point for your report, not a workaround.

## Report-ready material (already generated, in `outputs/`)

- `training_curves.png` — train vs. validation accuracy/loss across all
  16 training epochs, for your under/overfitting analysis
- `confusion_matrix.png` — 6×6 confusion matrix on the test set
- `classification_report.txt` — precision/recall/F1 per class + overall
  test accuracy
- `history.json` — raw per-epoch numbers behind the curves above

Quick read on the results: Frustrated, Confused, and Looking Away are
detected very reliably (F1 ≈ 0.94–0.96). Bored is the weakest class
(recall 0.58) — it gets confused with Drowsy sometimes, which makes
sense since both look visually similar (low energy, eyes semi-closed).
Worth a line in your report's limitations section.

## Project structure

```
project/
├── data/prepared/            # train/val/test images (2,277 total, ready to use)
├── models/
│   ├── best_model.keras      # ← the trained model the app actually uses
│   ├── final_model.keras
│   └── class_names.json      # exact class order — app/report must match this
├── pretrained/
│   └── mobilenet_v2_no_top.h5  # local ImageNet weights (no internet needed)
├── outputs/                  # training curves, confusion matrix, report, history.json
├── frontend/
│   └── index.html            # teacher dashboard (served by app.py)
├── src/
│   ├── prepare_data.py       # merge 3 raw datasets + split 70/15/15
│   ├── model.py              # MobileNetV2 transfer-learning architecture
│   ├── train.py              # training script (supports --resume)
│   ├── evaluate.py           # test-set evaluation (confusion matrix, report)
│   ├── face_detector.py      # OpenCV face detection/cropping
│   ├── inference.py          # loads model, predicts on face crops
│   └── app.py                # Flask backend + live camera + dashboard
└── requirements.txt
```
