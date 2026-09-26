"""
app.py
------
Flask backend for the Interactive Classroom Engagement dashboard.

Run:
    python src/app.py

This starts a local server, serves the dashboard at http://localhost:5000,
and automatically opens it in your browser. The dashboard shows the live
camera feed with each detected face labeled by predicted engagement
state, session timer controls, dynamic analytics, and end-of-session reports.
"""
import json
import os
import threading
import time
import webbrowser
from collections import Counter
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

from face_detector import detect_faces, crop_face
from inference import EngagementPredictor

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend"

app = Flask(__name__, static_folder=None)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ---------------------------------------------------------------------
# Global session state
# ---------------------------------------------------------------------
class SessionState:
    def __init__(self):
        self.active = False
        self.session_name = None
        self.started_at = None
        self.duration_limit = None  # seconds or None
        self.auto_stopped = False
        self.lock = threading.Lock()
        self.log = []               # list of {"t": timestamp, "label": str}
        self.latest_counts = Counter()  # counts within the current frame

    def start(self, name: str, duration_minutes: float = None):
        with self.lock:
            self.active = True
            self.session_name = name or f"Session {time.strftime('%Y-%m-%d %H:%M')}"
            self.started_at = time.time()
            self.duration_limit = (float(duration_minutes) * 60.0) if (duration_minutes and float(duration_minutes) > 0) else None
            self.auto_stopped = False
            self.log = []
            self.latest_counts = Counter()

    def check_auto_stop_locked(self):
        if self.active and self.duration_limit and self.started_at:
            elapsed = time.time() - self.started_at
            if elapsed >= self.duration_limit:
                self.active = False
                self.auto_stopped = True

    def stop(self):
        with self.lock:
            self.active = False
            summary = self._summary_locked()
        return summary

    def record(self, labels):
        """labels: list of predicted label strings for this frame."""
        with self.lock:
            self.check_auto_stop_locked()
            if not self.active:
                return
            now = time.time()
            self.latest_counts = Counter(labels)
            for label in labels:
                self.log.append({"t": now, "label": label})

    def snapshot(self):
        with self.lock:
            self.check_auto_stop_locked()
            elapsed = (time.time() - self.started_at) if self.started_at else 0.0
            remaining = None
            if self.duration_limit and self.started_at:
                remaining = max(0.0, self.duration_limit - elapsed)

            counts = Counter(e["label"] for e in self.log)
            total = sum(counts.values())
            
            # Engaged categories: Focused, Confused, Frustrated (active involvement)
            engaged_cnt = counts.get("Focused", 0) + counts.get("Confused", 0) + counts.get("Frustrated", 0)
            engagement_score = round(100.0 * engaged_cnt / total, 1) if total > 0 else 0.0

            return {
                "active": self.active,
                "auto_stopped": self.auto_stopped,
                "session_name": self.session_name,
                "elapsed_seconds": elapsed,
                "duration_limit": self.duration_limit,
                "remaining_seconds": remaining,
                "current_frame_counts": dict(self.latest_counts),
                "total_overall_counts": dict(counts),
                "num_datapoints": total,
                "engagement_score": engagement_score,
            }

    def _summary_locked(self):
        counts = Counter(e["label"] for e in self.log)
        total = sum(counts.values()) or 1
        elapsed = (time.time() - self.started_at) if self.started_at else 0.0

        percentages = {k: round(100.0 * v / total, 1) for k, v in counts.items()}

        # Core engagement metrics
        focused = percentages.get("Focused", 0.0)
        confused = percentages.get("Confused", 0.0)
        frustrated = percentages.get("Frustrated", 0.0)
        bored = percentages.get("Bored", 0.0)
        drowsy = percentages.get("Drowsy", 0.0)
        looking_away = percentages.get("Looking Away", 0.0)

        engaged_pct = round(focused + confused + frustrated, 1)

        # Classroom Health Rating
        if engaged_pct >= 75:
            health_rating = "Excellent Engagement"
            badge_color = "#3cb44b"
        elif engaged_pct >= 55:
            health_rating = "Good Attention"
            badge_color = "#4f8cff"
        elif engaged_pct >= 35:
            health_rating = "Moderate / Passive"
            badge_color = "#f0a500"
        else:
            health_rating = "Low Attention Needed"
            badge_color = "#e63946"

        # AI Teaching Insights
        insights = []
        if focused >= 50:
            insights.append("🌟 Class displays high attentiveness and strong absorption of the lecture material.")
        if bored + drowsy >= 30:
            insights.append("🥱 Fatigue or low energy detected in over 30% of student observations. A 2-minute energizer break or interactive poll is recommended.")
        if looking_away >= 25:
            insights.append("👀 Significant visual distraction noted. Try directing attention back to the whiteboard or main slides.")
        if confused >= 20:
            insights.append("🤔 Noticeable confusion detected. Pausing to ask for clarifying questions or re-explaining the current concept would be beneficial.")
        if frustrated >= 15:
            insights.append("⚡ Frustration signals observed. Consider breaking down complex topic steps or checking pacing.")
        if not insights:
            insights.append("👍 Class engagement remained steady throughout the session.")

        return {
            "session_name": self.session_name,
            "duration_seconds": elapsed,
            "num_datapoints": total,
            "counts": dict(counts),
            "percentages": percentages,
            "engagement_score": engaged_pct,
            "health_rating": health_rating,
            "badge_color": badge_color,
            "insights": insights,
        }


session_state = SessionState()

# Lazily initialized predictor
_predictor = None
_predictor_error = None


def get_predictor():
    global _predictor, _predictor_error
    if _predictor is None and _predictor_error is None:
        try:
            _predictor = EngagementPredictor(
                model_path=str(BASE_DIR / "models" / "best_model.joblib"),
                class_names_path=str(BASE_DIR / "models" / "class_names.json"),
            )
        except Exception as e:  # noqa: BLE001
            _predictor_error = str(e)
    return _predictor, _predictor_error


# Label Colors (BGR format for OpenCV drawing)
_LABEL_COLORS = {
    "Focused": (50, 205, 50),      # Lime / Emerald Green
    "Confused": (0, 191, 255),     # Amber / Deep Yellow
    "Frustrated": (70, 70, 255),    # Red-Orange
    "Bored": (211, 85, 186),       # Purple / Violet
    "Drowsy": (235, 130, 0),       # Cyan-Blue
    "Looking Away": (140, 140, 140)# Slate Gray
}


def box_color_for(label: str):
    return _LABEL_COLORS.get(label, (200, 200, 200))


def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Check camera permissions.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            faces = detect_faces(frame)
            predictor, err = get_predictor()

            frame_labels = []
            if faces and predictor is not None:
                crops = [crop_face(frame, box) for box in faces]
                crops = [c for c in crops if c.size > 0]
                results = predictor.predict_batch(crops)
                
                for (x, y, w, h), (label, conf, _) in zip(faces, results):
                    frame_labels.append(label)
                    color = box_color_for(label)

                    # Draw clean bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Label header pill background
                    text = f"{label} {conf * 100:.0f}%"
                    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    pill_y0 = max(0, y - th - 10)
                    cv2.rectangle(frame, (x, pill_y0), (x + tw + 10, pill_y0 + th + 8), color, -1)
                    
                    # White text overlay
                    cv2.putText(frame, text, (x + 5, pill_y0 + th + 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            elif err:
                cv2.putText(frame, "Model not loaded — see README",
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

            session_state.record(frame_labels)

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    finally:
        cap.release()


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/session/start", methods=["POST"])
def start_session():
    data = request.get_json(silent=True) or {}
    name = data.get("session_name", "").strip() or f"Session {time.strftime('%Y-%m-%d %H:%M')}"
    duration_minutes = data.get("duration_minutes", None)
    session_state.start(name, duration_minutes=duration_minutes)
    return jsonify({"ok": True, "session_name": name, "duration_minutes": duration_minutes})


@app.route("/api/session/stop", methods=["POST"])
def stop_session():
    summary = session_state.stop()
    return jsonify({"ok": True, "summary": summary})


@app.route("/api/stats")
def stats():
    return jsonify(session_state.snapshot())


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


def _open_browser():
    time.sleep(1.2)
    webbrowser.open("http://localhost:5000")


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
