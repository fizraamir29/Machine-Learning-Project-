"""
model.py
--------
Classical Machine Learning Feature Engineering Module for Student Engagement Detection.

================================================================================
MACHINE LEARNING CONCEPT 2: FEATURE ENGINEERING
================================================================================
In Deep Learning (CNNs), feature extraction is implicit and learned end-to-end via
convolutional filters, backpropagation, and millions of parameters.
In classical Machine Learning (SVM, Random Forest, KNN), we replace the neural network
entirely with DOMAIN-EXPERT FEATURE ENGINEERING:
  1. HOG (Histogram of Oriented Gradients) -> Captures edges, contours, and facial shape.
  2. LBP (Local Binary Patterns)          -> Captures fine micro-textures (skin, wrinkles, eyes).
  3. HSV Color Histograms                 -> Captures lighting-invariant color and complexion.
  4. Haar-Like Regional Differences       -> Captures facial landmark geometry and spatial contrast.

These 4 complementary representations are concatenated into a single fixed-length
feature vector (~938 dimensions) per face image, providing high discriminatory power
without deep neural networks or GPU acceleration.
"""

import os
from typing import List, Tuple
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

# Standardized face patch dimensions for feature extraction
IMG_SIZE: Tuple[int, int] = (96, 96)
NUM_CLASSES: int = 6
DEFAULT_CLASSES: List[str] = [
    "Bored",
    "Confused",
    "Drowsy",
    "Focused",
    "Frustrated",
    "Looking Away"
]

# Feature dimension breakdown:
# - HOG: 8 orientations x (2x2 cells/block) x (5x5 blocks) = 800 features
# - Spatial LBP: 3x3 grid x 10 uniform histogram bins       =  90 features
# - HSV Color Histograms: 16 (H) + 8 (S) + 8 (V)           =  32 features
# - Haar-Like Facial Region Differences                    =  16 features
# Total feature vector dimension                           = 938 features
FEATURE_DIM: int = 938


# ==============================================================================
# 1. HOG (Histogram of Oriented Gradients) Feature Extraction
# ==============================================================================
def extract_hog_features(gray_img: np.ndarray) -> np.ndarray:
    """
    Extracts Histogram of Oriented Gradients (HOG) features.
    
    ML Concept: Edge & Shape Representation
    - Evaluates local gradient magnitude and orientation distributions.
    - Robust against illumination variations due to block normalization (L2-Hys).
    - Captures the primary structural geometry of facial components (jawline, eyelids,
      eyebrow arches, mouth boundaries).
    """
    features = hog(
        gray_img,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )
    return features.astype(np.float32)


# ==============================================================================
# 2. LBP (Local Binary Patterns) Feature Extraction
# ==============================================================================
def extract_lbp_features(gray_img: np.ndarray, grid_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
    """
    Extracts Spatial Local Binary Pattern (LBP) texture histograms.
    
    ML Concept: Texture & Surface Microstructure Representation
    - Thresholds neighboring pixels against the center pixel to create binary codes.
    - Uses circular uniform patterns (P=8 points, radius R=1).
    - Uniform patterns have at most 2 bitwise transitions, creating 10 discrete bins:
      9 for uniform rotation patterns + 1 for non-uniform patterns.
    - Dividing the face into a 3x3 spatial grid preserves spatial locality
      (e.g., forehead texture vs eye texture vs cheek texture).
    """
    lbp = local_binary_pattern(gray_img, P=8, R=1, method="uniform")
    h, w = gray_img.shape
    gh, gw = grid_size
    step_y, step_x = h // gh, w // gw

    hist_list = []
    for r in range(gh):
        for c in range(gw):
            cell = lbp[r * step_y:(r + 1) * step_y, c * step_x:(c + 1) * step_x]
            hist, _ = np.histogram(cell.ravel(), bins=10, range=(0, 10), density=True)
            hist_list.append(hist)

    return np.concatenate(hist_list).astype(np.float32)


# ==============================================================================
# 3. Color Histograms (HSV Color Space)
# ==============================================================================
def extract_color_features(bgr_img: np.ndarray) -> np.ndarray:
    """
    Extracts normalized HSV color histograms.
    
    ML Concept: Illumination-Resilient Color Representation
    - Converts BGR to HSV (Hue, Saturation, Value) to decouple chromatic information (H)
      from lighting intensity (V).
    - Quantization: 16 bins for Hue, 8 bins for Saturation, 8 bins for Value.
    - Normalized by sum to produce probability distributions across color channels.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

    h_norm = h_hist / (np.sum(h_hist) + 1e-6)
    s_norm = s_hist / (np.sum(s_hist) + 1e-6)
    v_norm = v_hist / (np.sum(v_hist) + 1e-6)

    return np.concatenate([h_norm, s_norm, v_norm]).astype(np.float32)


# ==============================================================================
# 4. Haar-Like Regional Differences & Facial Landmark Geometry
# ==============================================================================
def extract_haar_like_features(gray_img: np.ndarray) -> np.ndarray:
    """
    Extracts Haar-like spatial contrast and landmark proxy features.
    
    ML Concept: Classical Viola-Jones Haar-like Contrast Features
    - Computes relative brightness differences between adjacent facial anatomical zones:
      * Forehead vs Eye region (brow furrowing, drowsiness indicator)
      * Nose vs Mouth region (mouth opening, yawning, yawning indicator)
      * Left face vs Right face (head turning, looking away indicator)
      * Sub-quadrant contrast and variance (overall facial symmetry)
    """
    h, w = gray_img.shape
    forehead   = gray_img[0:int(h * 0.25), :]
    eyes       = gray_img[int(h * 0.20):int(h * 0.50), :]
    nose       = gray_img[int(h * 0.40):int(h * 0.70), :]
    mouth      = gray_img[int(h * 0.65):int(h * 0.95), :]
    left_half  = gray_img[:, 0:int(w * 0.50)]
    right_half = gray_img[:, int(w * 0.50):]

    # Sub-quadrants
    mid_y, mid_x = h // 2, w // 2
    q1 = gray_img[0:mid_y, 0:mid_x]
    q2 = gray_img[0:mid_y, mid_x:]
    q3 = gray_img[mid_y:, 0:mid_x]
    q4 = gray_img[mid_y:, mid_x:]

    feats = [
        # Regional intensity differences
        float(np.mean(eyes)) - float(np.mean(forehead)),
        float(np.mean(mouth)) - float(np.mean(nose)),
        float(np.mean(left_half)) - float(np.mean(right_half)),
        # Regional standard deviations (texture / movement proxies)
        float(np.std(eyes)),
        float(np.std(mouth)),
        float(np.std(forehead)),
        float(np.std(left_half)) - float(np.std(right_half)),
        # Specific eye contrast
        float(np.mean(eyes)) / (float(np.mean(mouth)) + 1e-5),
        # 4 Quadrant Haar-like rectangular differences
        float(np.mean(q1)) - float(np.mean(q2)),
        float(np.mean(q3)) - float(np.mean(q4)),
        float(np.mean(q1)) - float(np.mean(q3)),
        float(np.mean(q2)) - float(np.mean(q4)),
        # Quadrant standard deviations
        float(np.std(q1)),
        float(np.std(q2)),
        float(np.std(q3)),
        float(np.std(q4)),
    ]
    return np.array(feats, dtype=np.float32)


# ==============================================================================
# Complete Classical ML Feature Vector Generator
# ==============================================================================
def extract_features(bgr_crop: np.ndarray, target_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Extracts the combined 938-dimensional feature vector from a face crop.
    
    Pipeline:
      Face Crop (BGR) -> Resize to (96, 96)
                      -> Grayscale -> HOG (800)
                      -> Grayscale -> LBP 3x3 Grid (90)
                      -> HSV       -> Color Histograms (32)
                      -> Grayscale -> Haar-like Features (16)
                      -> Concatenate -> Feature Vector (938-dim)
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    # Standardize image dimensions
    if bgr_crop.shape[:2] != target_size:
        bgr_crop = cv2.resize(bgr_crop, target_size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

    hog_feats   = extract_hog_features(gray)
    lbp_feats   = extract_lbp_features(gray)
    color_feats = extract_color_features(bgr_crop)
    haar_feats  = extract_haar_like_features(gray)

    feature_vector = np.concatenate([hog_feats, lbp_feats, color_feats, haar_feats])
    return feature_vector.astype(np.float32)


def get_feature_names() -> List[str]:
    """
    Returns human-interpretable feature names for all 938 dimensions.
    Used for Random Forest Feature Importance visualization and model explainability.
    """
    names = []
    # 800 HOG features
    names.extend([f"HOG_block_{i}" for i in range(800)])
    # 90 Spatial LBP features
    for grid_idx in range(9):
        names.extend([f"LBP_grid{grid_idx}_bin_{b}" for b in range(10)])
    # 32 Color features
    names.extend([f"HSV_Hue_bin_{i}" for i in range(16)])
    names.extend([f"HSV_Sat_bin_{i}" for i in range(8)])
    names.extend([f"HSV_Val_bin_{i}" for i in range(8)])
    # 16 Haar-like features
    names.extend([
        "Haar_Eyes_vs_Forehead",
        "Haar_Mouth_vs_Nose",
        "Haar_Left_vs_Right_Mean",
        "Haar_Eyes_StdDev",
        "Haar_Mouth_StdDev",
        "Haar_Forehead_StdDev",
        "Haar_Left_vs_Right_StdDev",
        "Haar_Eyes_to_Mouth_Ratio",
        "Haar_Q1_vs_Q2_HorizTop",
        "Haar_Q3_vs_Q4_HorizBottom",
        "Haar_Q1_vs_Q3_VertLeft",
        "Haar_Q2_vs_Q4_VertRight",
        "Haar_Q1_StdDev",
        "Haar_Q2_StdDev",
        "Haar_Q3_StdDev",
        "Haar_Q4_StdDev",
    ])
    return names


if __name__ == "__main__":
    dummy_face = np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8)
    feats = extract_features(dummy_face)
    print(f"Feature vector shape: {feats.shape} (Expected: {FEATURE_DIM})")
    print(f"Total feature names generated: {len(get_feature_names())}")
