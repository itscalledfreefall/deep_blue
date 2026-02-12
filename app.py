#!/usr/bin/env python3
"""
Deep Blue Web - Forklift Safety System Dashboard
Unified Flask app: camera capture, YOLO detection, MJPEG streaming,
ROI zone drawing, GPIO relay control, and web login.
"""

import json
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, session, url_for)
from picamera2 import Picamera2
from ultralytics import YOLO

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# --- Configuration ---
MODEL_PATH = "/home/enigma/yolo26n_256.onnx"
IMGSZ = 256
CONF_THRESHOLD = 0.25
CAM_WIDTH = 640
CAM_HEIGHT = 480
PERSON_CLASS = 0
RELAY_1_PIN = 23
RELAY_2_PIN = 24
TRIGGER_THRESHOLD = 2
RELEASE_SECONDS = 1.5

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
SECRET_KEY = "deepblue-forklift-safety-2026"
USERNAME = "deepblue"
PASSWORD = "matrix18"

app = Flask(__name__)
app.secret_key = SECRET_KEY

# --- Shared State ---
lock = threading.Lock()
latest_frame = None        # raw camera frame (RGB)
annotated_frame = None     # frame with bounding boxes + ROI overlay (BGR for JPEG)
roi_polygon = []           # list of [x, y] points
status = {
    "fps": 0.0,
    "detections": 0,
    "relay": "OFF",
    "running": False,
}


def load_config():
    global roi_polygon
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            roi_polygon = data.get("roi", [])
        except Exception:
            roi_polygon = []


def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump({"roi": roi_polygon}, f)


# --- GPIO ---
def setup_gpio():
    if not GPIO_AVAILABLE:
        print("WARNING: RPi.GPIO not available, relays disabled")
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(RELAY_1_PIN, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(RELAY_2_PIN, GPIO.OUT, initial=GPIO.HIGH)
    print(f"Relays: BCM {RELAY_1_PIN} + BCM {RELAY_2_PIN} (active-low)")


relay_active = False


def set_relay(active):
    global relay_active
    if relay_active == active:
        return
    relay_active = active
    status["relay"] = "ON" if active else "OFF"
    if GPIO_AVAILABLE:
        state = GPIO.LOW if active else GPIO.HIGH
        GPIO.output(RELAY_1_PIN, state)
        GPIO.output(RELAY_2_PIN, state)


# --- Camera Thread ---
def camera_thread():
    global latest_frame
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    print(f"Camera started: {CAM_WIDTH}x{CAM_HEIGHT}")

    while status["running"]:
        frame = picam2.capture_array().copy()  # copy to prevent buffer reuse
        with lock:
            latest_frame = frame
        time.sleep(0.01)

    picam2.stop()
    print("Camera stopped")


# --- YOLO Detection Thread ---
def detection_thread():
    global annotated_frame
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")

    # Wait for first frame
    while status["running"]:
        with lock:
            f = latest_frame
        if f is not None:
            break
        time.sleep(0.1)

    # Warmup
    if f is not None:
        for _ in range(3):
            model(f, imgsz=IMGSZ, verbose=False)
    print("YOLO warmup done")

    consecutive_detections = 0
    last_detection_time = 0

    while status["running"]:
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.05)
            continue

        t0 = time.time()
        current_roi = roi_polygon[:]

        # Always run YOLO on the full frame for best detection
        results = model(frame, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)

        all_persons = [b for b in results[0].boxes if int(b.cls) == PERSON_CLASS]

        # Filter: only persons whose bbox center is inside the ROI polygon
        if current_roi and len(current_roi) >= 3:
            roi_contour = np.array(current_roi, dtype=np.int32)
            persons_in_roi = []
            persons_outside = []
            for b in all_persons:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if cv2.pointPolygonTest(roi_contour, (float(cx), float(cy)), False) >= 0:
                    persons_in_roi.append(b)
                else:
                    persons_outside.append(b)
        else:
            # No ROI = full frame counts
            persons_in_roi = all_persons
            persons_outside = []

        elapsed = time.time() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        now = time.time()

        if persons_in_roi:
            consecutive_detections += 1
            last_detection_time = now
            if consecutive_detections >= TRIGGER_THRESHOLD:
                set_relay(True)
        else:
            consecutive_detections = 0
            if relay_active and (now - last_detection_time) >= RELEASE_SECONDS:
                set_relay(False)

        # Annotate frame for streaming (convert RGB to BGR for cv2 drawing)
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw ROI zone
        if current_roi and len(current_roi) >= 3:
            pts = np.array(current_roi, dtype=np.int32)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 180, 0))
            cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)

        # Draw bounding boxes - red for in-ROI, gray for outside
        for box in persons_in_roi:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display, f"PERSON {conf:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for box in persons_outside:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(display, (x1, y1), (x2, y2), (128, 128, 128), 1)

        # Draw status bar
        relay_color = (0, 0, 255) if relay_active else (0, 200, 0)
        relay_text = "ALARM ON" if relay_active else "CLEAR"
        cv2.putText(display, f"FPS: {fps:.1f} | {relay_text} | Persons: {len(persons_in_roi)}",
                    (10, CAM_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, relay_color, 2)

        with lock:
            annotated_frame = display

        status["fps"] = round(fps, 1)
        status["detections"] = len(persons_in_roi)

    print("Detection stopped")


# --- MJPEG Generator ---
def mjpeg_generator():
    while True:
        with lock:
            frame = annotated_frame.copy() if annotated_frame is not None else None
        if frame is None:
            time.sleep(0.05)
            continue
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.05)  # ~20fps stream rate to reduce Pi load


# --- Auth ---
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# --- Routes ---
@app.route("/")
def index():
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if (request.form.get("username") == USERNAME and
                request.form.get("password") == PASSWORD):
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        error = "Invalid credentials"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/roi", methods=["GET"])
@login_required
def get_roi():
    return jsonify({"roi": roi_polygon})


@app.route("/api/roi", methods=["POST"])
@login_required
def set_roi():
    global roi_polygon
    data = request.get_json(force=True)
    roi_polygon = data.get("roi", [])
    save_config()
    return jsonify({"ok": True, "roi": roi_polygon})


@app.route("/api/status")
@login_required
def get_status():
    return jsonify(status)


# --- Main ---
def shutdown_handler(signum, frame):
    status["running"] = False
    if GPIO_AVAILABLE:
        GPIO.output(RELAY_1_PIN, GPIO.HIGH)
        GPIO.output(RELAY_2_PIN, GPIO.HIGH)
        GPIO.cleanup()
    print("Deep Blue Web shutting down...")
    sys.exit(0)


if __name__ == "__main__":
    load_config()
    setup_gpio()
    status["running"] = True

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    cam_t = threading.Thread(target=camera_thread, daemon=True)
    det_t = threading.Thread(target=detection_thread, daemon=True)
    cam_t.start()
    det_t.start()

    print("Starting Deep Blue Web on http://0.0.0.0:80")
    app.run(host="0.0.0.0", port=80, threaded=True, debug=False)
