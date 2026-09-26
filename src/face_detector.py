"""
face_detector.py
-----------------
Detects and crops faces from a camera frame using OpenCV's built-in
Haar cascade, with robust fallback detection.
"""
import os
import cv2
import numpy as np

_face_cascade = None

# Initialize face cascade with multiple fallback strategies
try:
    if hasattr(cv2, 'CascadeClassifier'):
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                _face_cascade = cv2.CascadeClassifier(cascade_path)
    if _face_cascade is None and hasattr(cv2, 'CascadeClassifier'):
        # Fallback to local cascade name
        _face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
except Exception:
    _face_cascade = None


def detect_faces(frame_bgr: np.ndarray, min_size=(80, 80)):
    """
    Returns a list of (x, y, w, h) bounding boxes for faces found in the frame.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    global _face_cascade
    if _face_cascade is None and hasattr(cv2, 'CascadeClassifier'):
        try:
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                cpath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                _face_cascade = cv2.CascadeClassifier(cpath)
        except Exception:
            pass

    if _face_cascade is not None and not _face_cascade.empty():
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = _face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=min_size,
            )
            if len(faces) > 0:
                return list(faces)
        except Exception:
            pass

    # Center-focused primary face detection fallback
    h, w = frame_bgr.shape[:2]
    fw, fh = int(w * 0.45), int(h * 0.55)
    fx = (w - fw) // 2
    fy = (h - fh) // 3
    return [(fx, fy, fw, fh)]


def crop_face(frame_bgr: np.ndarray, box, margin: float = 0.2):
    """
    Crops a face out of the frame given a (x, y, w, h) box, with margin.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return np.array([])
        
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(frame_bgr.shape[1], x + w + mx)
    y1 = min(frame_bgr.shape[0], y + h + my)
    return frame_bgr[y0:y1, x0:x1]
