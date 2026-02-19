#!/usr/bin/env python3
"""
Deep Blue Web - Forklift Safety System Dashboard
Unified Flask app: camera capture, YOLO detection, MJPEG streaming,
ROI zone drawing, GPIO relay control, and web login.
"""

import json
import os
import re
import signal
import sys
import threading
import time

# ONNX Runtime thread config (must be set before import)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, session, url_for)
from ultralytics import YOLO

PICAMERA2_IMPORT_ERROR = None
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception as e:
    PICAMERA2_IMPORT_ERROR = str(e)
    PICAMERA2_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# --- Configuration ---
# Default to NCNN FP16 export for better Raspberry Pi performance.
# Override at runtime with env var MODEL_PATH for quick A/B testing.
MODEL_PATH = os.getenv("MODEL_PATH", "/home/enigma/yolo26n_ncnn_fp16")
FALLBACK_MODEL_PATH = "/home/enigma/yolo26n_256.onnx"
# Keep imgsz aligned with exported model size.
IMGSZ = 256
# Temporary performance profile: set to False to restore richer overlays/colors.
PERFORMANCE_MODE = True
CONF_THRESHOLD = 0.35 if PERFORMANCE_MODE else 0.20
CAM_WIDTH = 640
CAM_HEIGHT = 480
# Compensate wide-angle lenses by center-cropping then resizing back.
# 1.0 = no zoom (cleanest), 1.2~1.5 increases apparent zoom but softens image.
try:
    DIGITAL_ZOOM = float(os.getenv("DIGITAL_ZOOM", "1.0"))
except ValueError:
    DIGITAL_ZOOM = 1.0
if DIGITAL_ZOOM < 1.0:
    DIGITAL_ZOOM = 1.0
# Default to Raspberry Pi CSI camera (IMX via Picamera2).
CAMERA_BACKEND = os.getenv("CAMERA_BACKEND", "picamera2").lower()  # auto|usb|picamera2
try:
    USB_CAMERA_INDEX = int(os.getenv("USB_CAMERA_INDEX", "0"))
except ValueError:
    USB_CAMERA_INDEX = 0
USB_CAPTURE_FOURCC = os.getenv("USB_CAPTURE_FOURCC", "YUYV").upper()
# Enable NoIR tuning by default for IMX219-160IR modules to reduce daytime color cast.
PICAMERA2_USE_NOIR_TUNING = os.getenv("PICAMERA2_USE_NOIR_TUNING", "1").lower() not in {
    "0", "false", "no", "off"
}
PICAMERA2_TUNING_FILE = os.getenv("PICAMERA2_TUNING_FILE", "").strip()
PICAMERA2_NOIR_TUNING_CANDIDATES = [
    # Pi 5 (pisp)
    "/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json",
    # Pi 4 and earlier (vc4)
    "/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json",
    # Older libcamera layout
    "/usr/share/libcamera/ipa/raspberrypi/imx219_noir.json",
]
CAMERA_READ_SLEEP = 0.005 if PERFORMANCE_MODE else 0.01
CAMERA_RETRY_DELAY = 1.0
CAMERA_MAX_READ_FAILURES = 30
# Daytime color tuning for IR-sensitive cameras (e.g., IMX219-160IR/NoIR).
# These are best-effort controls; unsupported controls are skipped safely.
CAMERA_DAY_CONTROLS = {
    "AeEnable": True,
    "AwbEnable": True,
    "AwbMode": 0,      # Auto
    "Contrast": 1.10,
    "Saturation": 0.75
}
# Additional software color correction (R, G, B gains) to reduce magenta cast.
# Disabled in performance mode to avoid per-frame float processing overhead.
RGB_COLOR_GAINS = (1.0, 1.0, 1.0) if PERFORMANCE_MODE else (0.78, 1.00, 1.08)
# Keep zone interiors visible for accurate operator feedback.
DRAW_ZONE_FILL = os.getenv("DRAW_ZONE_FILL", "1").lower() not in {"0", "false", "no", "off"}
DRAW_OUTSIDE_BOXES = not PERFORMANCE_MODE
DRAW_BOX_LABELS = not PERFORMANCE_MODE
MJPEG_QUALITY = 72 if PERFORMANCE_MODE else 70
MJPEG_STREAM_SLEEP = 0.03 if PERFORMANCE_MODE else 0.05
PERSON_CLASS = 0
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
RELAY_1_PIN = 23
RELAY_2_PIN = 24
GREEN_TRIGGER_THRESHOLD = 2  # anti-flicker: require N consecutive vehicle-only frames
# Camera-motion detection for vehicle_road zones (pixel-change based)
MOTION_BG_HISTORY = 300
MOTION_BG_VAR_THRESHOLD = 40
MOTION_PIXEL_THRESHOLD = 0.04   # 4% of zone pixels must change
MOTION_CONSECUTIVE_FRAMES = 2   # anti-flicker
MOTION_WARMUP_FRAMES = 30       # ignore first N frames while MOG2 builds background model
try:
    VEHICLE_EXIT_GRACE_SECONDS = float(os.getenv("VEHICLE_EXIT_GRACE_SECONDS", "1.2"))
except ValueError:
    VEHICLE_EXIT_GRACE_SECONDS = 1.2
if VEHICLE_EXIT_GRACE_SECONDS < 0:
    VEHICLE_EXIT_GRACE_SECONDS = 0.0

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
SECRET_KEY = "deepblue-forklift-safety-2026"
USERNAME = "deepblue"
PASSWORD = "matrix18"
DEFAULT_ZONE_COLOR = "#00e5ff"
HEX_COLOR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")

app = Flask(__name__)
app.secret_key = SECRET_KEY

# --- Shared State ---
lock = threading.Lock()
latest_frame = None        # raw camera frame (RGB)
annotated_frame = None     # frame with bounding boxes + ROI overlay (BGR for JPEG)
zones = []                 # list of {"id", "name", "color", "zone_type", "points"}
MAX_ZONES = 8
status = {
    "fps": 0.0,
    "persons": 0,
    "vehicles": 0,
    "motion_zones": 0,
    "relay": "OFF",
    "running": False,
}

# Camera-motion detection shared state
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=MOTION_BG_HISTORY,
    varThreshold=MOTION_BG_VAR_THRESHOLD,
    detectShadows=False
)
motion_state = {}       # zone_id -> {consecutive_frames, last_detected_ts}
motion_frame_count = 0  # tracks frames since last MOG2 reset for warmup
camera_reconnected = False  # flag set by camera thread on reconnect


def load_config():
    global zones
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            if "zones" in data:
                zones = data["zones"]
                # Migrate zones missing zone_type
                for z in zones:
                    z["color"] = normalize_zone_color(z.get("color", DEFAULT_ZONE_COLOR))
                    if "zone_type" not in z:
                        z["zone_type"] = "human"
            elif "roi" in data and data["roi"]:
                # Migrate old single-zone format
                zones = [{
                    "id": "z_legacy",
                    "name": "Zone 1",
                    "color": DEFAULT_ZONE_COLOR,
                    "zone_type": "human",
                    "points": data["roi"]
                }]
                save_config()
            else:
                zones = []
        except Exception:
            zones = []


def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump({"zones": zones}, f)


def normalize_zone_color(color, fallback=DEFAULT_ZONE_COLOR):
    if isinstance(color, str):
        c = color.strip()
        if HEX_COLOR_PATTERN.fullmatch(c):
            return c.lower()
    return fallback


def hex_to_bgr(hex_color):
    h = normalize_zone_color(hex_color).lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


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


def apply_camera_controls(picam2, controls):
    for key, value in controls.items():
        try:
            picam2.set_controls({key: value})
        except Exception:
            # Keep startup resilient across different camera/driver stacks.
            pass


def apply_rgb_color_gains(frame, gains):
    if gains == (1.0, 1.0, 1.0):
        return frame
    fr = frame.astype(np.float32)
    fr[..., 0] *= gains[0]  # R
    fr[..., 1] *= gains[1]  # G
    fr[..., 2] *= gains[2]  # B
    return np.clip(fr, 0, 255).astype(np.uint8)


def apply_digital_zoom(frame, zoom):
    if zoom <= 1.0:
        return frame

    h, w = frame.shape[:2]
    crop_w = max(2, int(w / zoom))
    crop_h = max(2, int(h / zoom))
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    crop = frame[y0:y0 + crop_h, x0:x0 + crop_w]
    if crop.size == 0:
        return frame
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def detect_motion_in_zone(fg_mask, zone_points, frame_shape):
    """Check if significant pixel change detected in camera feed within zone polygon."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(zone_points, dtype=np.int32)], 255)
    zone_fg = cv2.bitwise_and(fg_mask, mask)
    zone_area = cv2.countNonZero(mask)
    if zone_area == 0:
        return False
    return cv2.countNonZero(zone_fg) / zone_area >= MOTION_PIXEL_THRESHOLD


def create_picamera2_instance():
    if not PICAMERA2_AVAILABLE:
        return None

    # Explicit override takes priority.
    if PICAMERA2_TUNING_FILE:
        if os.path.exists(PICAMERA2_TUNING_FILE):
            try:
                tuning = Picamera2.load_tuning_file(PICAMERA2_TUNING_FILE)
                print(f"Picamera2 tuning file: {PICAMERA2_TUNING_FILE}")
                return Picamera2(tuning=tuning)
            except Exception as e:
                print(f"WARNING: failed to load tuning file {PICAMERA2_TUNING_FILE}: {e}")
        else:
            print(f"WARNING: tuning file not found: {PICAMERA2_TUNING_FILE}")

    if PICAMERA2_USE_NOIR_TUNING:
        for path in PICAMERA2_NOIR_TUNING_CANDIDATES:
            if not os.path.exists(path):
                continue
            try:
                tuning = Picamera2.load_tuning_file(path)
                print(f"Picamera2 tuning file: {path}")
                return Picamera2(tuning=tuning)
            except Exception as e:
                print(f"WARNING: failed to load tuning file {path}: {e}")

    print("Picamera2 tuning file: default")
    return Picamera2()


def resolve_model_path(path):
    if os.path.isdir(path):
        direct_param = os.path.join(path, "model.ncnn.param")
        direct_bin = os.path.join(path, "model.ncnn.bin")
        if os.path.exists(direct_param) and os.path.exists(direct_bin):
            # Ultralytics NCNN backend expects the model directory, not .param file path.
            return path

        # Some exports are nested one level deeper, e.g. <dir>/*_ncnn_model/model.ncnn.param
        try:
            for name in sorted(os.listdir(path)):
                nested_dir = os.path.join(path, name)
                if not os.path.isdir(nested_dir):
                    continue
                nested_param = os.path.join(nested_dir, "model.ncnn.param")
                nested_bin = os.path.join(nested_dir, "model.ncnn.bin")
                if os.path.exists(nested_param) and os.path.exists(nested_bin):
                    return nested_dir
        except Exception:
            pass

    return path


def decode_fourcc(value):
    v = int(value)
    return "".join([chr((v >> (8 * i)) & 0xFF) for i in range(4)])


def camera_thread_usb():
    global latest_frame, camera_reconnected
    cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(USB_CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: USB camera /dev/video{USB_CAMERA_INDEX} not available")
        with lock:
            latest_frame = None
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if len(USB_CAPTURE_FOURCC) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*USB_CAPTURE_FOURCC))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    print(
        f"USB camera started: /dev/video{USB_CAMERA_INDEX} "
        f"({actual_w}x{actual_h}, FOURCC={actual_fourcc})"
    )

    # Prime capture and skip initial unstable frames from some USB devices.
    for _ in range(8):
        cap.read()
        time.sleep(0.01)

    camera_reconnected = True
    restart_required = False
    read_failures = 0
    while status["running"]:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            read_failures += 1
            if read_failures >= CAMERA_MAX_READ_FAILURES:
                print("WARNING: USB camera read failed repeatedly, restarting camera")
                restart_required = True
                break
            time.sleep(0.02)
            continue
        read_failures = 0
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = apply_digital_zoom(frame, DIGITAL_ZOOM)
        with lock:
            latest_frame = frame
        time.sleep(CAMERA_READ_SLEEP)

    cap.release()
    if restart_required:
        with lock:
            latest_frame = None
        return False
    print("USB camera stopped")
    return True


def camera_thread_picamera2():
    global latest_frame, camera_reconnected
    if not PICAMERA2_AVAILABLE:
        print(f"ERROR: Picamera2 is not available: {PICAMERA2_IMPORT_ERROR}")
        with lock:
            latest_frame = None
        return False

    try:
        picam2 = create_picamera2_instance()
        if picam2 is None:
            print(f"ERROR: Picamera2 is not available: {PICAMERA2_IMPORT_ERROR}")
            with lock:
                latest_frame = None
            return False
        config = picam2.create_preview_configuration(
            main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        apply_camera_controls(picam2, CAMERA_DAY_CONTROLS)
    except Exception as e:
        print(f"ERROR: Picamera2 start failed: {e}")
        with lock:
            latest_frame = None
        return False

    time.sleep(1)
    print(f"Picamera2 started: {CAM_WIDTH}x{CAM_HEIGHT}")

    camera_reconnected = True
    restart_required = False
    read_failures = 0
    while status["running"]:
        try:
            frame = picam2.capture_array().copy()  # copy to prevent buffer reuse
        except Exception:
            read_failures += 1
            if read_failures >= CAMERA_MAX_READ_FAILURES:
                print("WARNING: Picamera2 capture failed repeatedly, restarting camera")
                restart_required = True
                break
            time.sleep(0.05)
            continue
        read_failures = 0
        frame = apply_rgb_color_gains(frame, RGB_COLOR_GAINS)
        frame = apply_digital_zoom(frame, DIGITAL_ZOOM)
        with lock:
            latest_frame = frame
        time.sleep(CAMERA_READ_SLEEP)

    try:
        picam2.stop()
    except Exception:
        pass
    try:
        picam2.close()
    except Exception:
        pass

    if restart_required:
        with lock:
            latest_frame = None
        return False
    print("Picamera2 stopped")
    return True


# --- Camera Thread ---
def camera_thread():
    backend = CAMERA_BACKEND
    if backend not in {"auto", "usb", "picamera2"}:
        print(f"WARNING: invalid CAMERA_BACKEND={backend}, using auto")
        backend = "auto"

    print(f"Camera backend mode: {backend}")

    while status["running"]:
        if backend == "usb":
            if camera_thread_usb():
                return
            print(f"WARNING: retrying USB camera in {CAMERA_RETRY_DELAY:.1f}s")
            time.sleep(CAMERA_RETRY_DELAY)
            continue

        if backend == "picamera2":
            if camera_thread_picamera2():
                return
            print(f"WARNING: retrying Picamera2 camera in {CAMERA_RETRY_DELAY:.1f}s")
            time.sleep(CAMERA_RETRY_DELAY)
            continue

        # auto: prefer USB first, then fallback to Picamera2
        if camera_thread_usb():
            return
        if not status["running"]:
            return
        print("WARNING: USB camera unavailable, trying Picamera2")
        if camera_thread_picamera2():
            return
        if not status["running"]:
            return
        print(f"ERROR: no camera source available; retrying in {CAMERA_RETRY_DELAY:.1f}s")
        time.sleep(CAMERA_RETRY_DELAY)


# --- YOLO Detection Thread ---
def detection_thread():
    global annotated_frame, bg_subtractor, motion_frame_count, camera_reconnected
    def model_candidates():
        candidates = []
        for raw_path in (MODEL_PATH, FALLBACK_MODEL_PATH):
            resolved = resolve_model_path(raw_path)
            if resolved and resolved not in candidates and os.path.exists(resolved):
                candidates.append(resolved)
        return candidates

    def load_model_with_warmup(warmup_frame):
        candidates = model_candidates()
        if not candidates:
            print("ERROR: no model candidates found")
            return None

        last_error = None
        for i, candidate in enumerate(candidates):
            if i == 0:
                print(f"Loading model: {candidate}")
            else:
                print(f"Falling back to model: {candidate}")
            try:
                loaded_model = YOLO(candidate, task="detect")
                if warmup_frame is not None:
                    for _ in range(3):
                        loaded_model(warmup_frame, imgsz=IMGSZ, verbose=False)
                return loaded_model
            except Exception as e:
                last_error = e
                print(f"WARNING: model init/warmup failed for '{candidate}': {e}")

        print(f"ERROR: all model candidates failed: {last_error}")
        return None

    # Wait for first frame
    f = None
    while status["running"]:
        with lock:
            f = latest_frame
        if f is not None:
            break
        time.sleep(0.1)

    model = load_model_with_warmup(f)
    if model is None:
        if relay_active:
            set_relay(False)
        while status["running"]:
            time.sleep(0.5)
        return

    print("YOLO warmup done")

    green_candidate_frames = 0
    last_vehicle_seen_ts = 0.0

    while status["running"]:
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.05)
            continue

        t0 = time.time()
        current_zones = [z for z in zones if len(z.get("points", [])) >= 3]

        # Run YOLO on full frame
        try:
            results = model(frame, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)
        except Exception as e:
            print(f"WARNING: inference failed, reloading model: {e}")
            model = load_model_with_warmup(frame)
            if model is None:
                if relay_active:
                    set_relay(False)
                while status["running"]:
                    time.sleep(0.5)
                return
            continue

        # Reset MOG2 on camera reconnect
        if camera_reconnected:
            camera_reconnected = False
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=MOTION_BG_HISTORY,
                varThreshold=MOTION_BG_VAR_THRESHOLD,
                detectShadows=False
            )
            motion_frame_count = 0
            motion_state.clear()

        # Separate persons and vehicles
        all_persons = []
        all_vehicles = []
        for b in results[0].boxes:
            cls_id = int(b.cls)
            if cls_id == PERSON_CLASS:
                all_persons.append(b)
            elif cls_id in VEHICLE_CLASSES:
                all_vehicles.append(b)

        # Filter by ROI: only count detections whose bbox center is inside
        def filter_by_roi(boxes, roi_contour):
            inside, outside = [], []
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if cv2.pointPolygonTest(roi_contour, (float(cx), float(cy)), False) >= 0:
                    inside.append(b)
                else:
                    outside.append(b)
            return inside, outside

        # Camera-motion detection: apply MOG2 to grayscale frame
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        fg_mask = bg_subtractor.apply(gray)
        motion_frame_count += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        motion_warmup_done = motion_frame_count > MOTION_WARMUP_FRAMES

        # Zone-type-aware filtering
        human_zones = []
        road_zones = []
        for zone in current_zones:
            if zone.get("zone_type", "human") == "vehicle_road":
                road_zones.append(zone)
            else:
                human_zones.append(zone)

        # Human zones: YOLO person detection only
        persons_in_roi = []
        persons_outside = list(all_persons)
        vehicles_in_roi = []
        vehicles_outside = list(all_vehicles)
        if human_zones:
            for zone in human_zones:
                roi_contour = np.array(zone["points"], dtype=np.int32)
                p_in, persons_outside = filter_by_roi(persons_outside, roi_contour)
                persons_in_roi.extend(p_in)
                v_in, vehicles_outside = filter_by_roi(vehicles_outside, roi_contour)
                vehicles_in_roi.extend(v_in)
        elif not current_zones:
            # No zones at all: full frame fallback
            persons_in_roi, persons_outside = all_persons, []
            vehicles_in_roi, vehicles_outside = all_vehicles, []

        # Vehicle road zones: camera-motion detection
        motion_zones_active = 0
        active_motion_zone_ids = set()
        if road_zones and motion_warmup_done:
            for zone in road_zones:
                zid = zone["id"]
                has_motion = detect_motion_in_zone(fg_mask, zone["points"], frame.shape)
                if zid not in motion_state:
                    motion_state[zid] = {"consecutive_frames": 0, "last_detected_ts": 0.0}
                if has_motion:
                    motion_state[zid]["consecutive_frames"] += 1
                    if motion_state[zid]["consecutive_frames"] >= MOTION_CONSECUTIVE_FRAMES:
                        motion_state[zid]["last_detected_ts"] = time.time()
                        motion_zones_active += 1
                        active_motion_zone_ids.add(zid)
                else:
                    motion_state[zid]["consecutive_frames"] = 0

        # Clean stale motion_state entries
        active_zone_ids = {z["id"] for z in current_zones}
        for zid in list(motion_state):
            if zid not in active_zone_ids:
                del motion_state[zid]

        elapsed = time.time() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        # Traffic logic:
        # - Person in human zone: force relay OFF immediately (highest priority).
        # - Vehicle in human zone OR motion in vehicle_road zone: relay ON (anti-flicker).
        # - No detections: relay OFF after grace window.
        now = time.time()
        vehicle_present = len(vehicles_in_roi) > 0 or motion_zones_active > 0
        pedestrian_present = len(persons_in_roi) > 0
        if vehicle_present:
            last_vehicle_seen_ts = now

        if pedestrian_present:
            green_candidate_frames = 0
            if relay_active:
                set_relay(False)
        elif vehicle_present:
            green_candidate_frames += 1
            if green_candidate_frames >= GREEN_TRIGGER_THRESHOLD:
                set_relay(True)
        else:
            green_candidate_frames = 0
            vehicle_recent = (now - last_vehicle_seen_ts) <= VEHICLE_EXIT_GRACE_SECONDS
            if relay_active and not vehicle_recent:
                set_relay(False)

        # Annotate frame for streaming (convert RGB to BGR for cv2 drawing)
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw all zones with their colors
        if current_zones:
            if DRAW_ZONE_FILL:
                overlay = display.copy()
                for zone in current_zones:
                    pts = np.array(zone["points"], dtype=np.int32)
                    bgr = hex_to_bgr(zone.get("color", "#00e5ff"))
                    # Stronger fill for vehicle_road zones with active motion
                    alpha = 0.2 if zone["id"] in active_motion_zone_ids else 0.08
                    cv2.fillPoly(overlay, [pts], bgr)
                    cv2.addWeighted(overlay, alpha, display, 1.0 - alpha, 0, display)
                    overlay = display.copy()
            for zone in current_zones:
                pts = np.array(zone["points"], dtype=np.int32)
                bgr = hex_to_bgr(zone.get("color", "#00e5ff"))
                cv2.polylines(display, [pts], True, bgr, 2)
                name = zone.get("name", "")
                if DRAW_BOX_LABELS and name:
                    (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cx = int(np.mean(pts[:, 0])) - tw // 2
                    cy = int(np.mean(pts[:, 1])) + th // 2
                    cv2.putText(display, name, (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                    cv2.putText(display, name, (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw person boxes - red in ROI, gray outside
        for box in persons_in_roi:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if DRAW_BOX_LABELS:
                cv2.putText(display, f"PERSON {conf:.0%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if DRAW_OUTSIDE_BOXES:
            for box in persons_outside:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(display, (x1, y1), (x2, y2), (128, 128, 128), 1)

        # Draw vehicle boxes - yellow in ROI, gray outside
        for box in vehicles_in_roi:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = VEHICLE_CLASSES.get(cls_id, "vehicle").upper()
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 255), 2)
            if DRAW_BOX_LABELS:
                cv2.putText(display, f"{label} {conf:.0%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
        if DRAW_OUTSIDE_BOXES:
            for box in vehicles_outside:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(display, (x1, y1), (x2, y2), (128, 128, 128), 1)

        # Draw status bar
        relay_color = (0, 200, 0) if relay_active else (0, 0, 255)
        relay_text = "GREEN" if relay_active else "RED"
        status_text = f"FPS: {fps:.1f} | {relay_text} | P:{len(persons_in_roi)} V:{len(vehicles_in_roi)}"
        if motion_zones_active > 0:
            status_text += f" M:{motion_zones_active}"
        cv2.putText(display, status_text,
                    (10, CAM_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, relay_color, 2)

        with lock:
            annotated_frame = display

        status["fps"] = round(fps, 1)
        status["persons"] = len(persons_in_roi)
        status["vehicles"] = len(vehicles_in_roi)
        status["motion_zones"] = motion_zones_active

    print("Detection stopped")


# --- MJPEG Generator ---
def mjpeg_generator():
    while True:
        with lock:
            frame = annotated_frame.copy() if annotated_frame is not None else None
        if frame is None:
            time.sleep(MJPEG_STREAM_SLEEP)
            continue
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(MJPEG_STREAM_SLEEP)


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


@app.route("/api/zones", methods=["GET"])
@login_required
def get_zones():
    return jsonify({"zones": zones})


@app.route("/api/zones", methods=["POST"])
@login_required
def add_zone():
    global zones
    if len(zones) >= MAX_ZONES:
        return jsonify({"ok": False, "error": "Max 8 zones"}), 400
    data = request.get_json(force=True)
    zone_type = data.get("zone_type", "human")
    if zone_type not in ("human", "vehicle_road"):
        zone_type = "human"
    zone = {
        "id": data.get("id", f"z{int(time.time() * 1000)}"),
        "name": data.get("name", f"Zone {len(zones) + 1}"),
        "color": normalize_zone_color(data.get("color")),
        "zone_type": zone_type,
        "points": data.get("points", []),
    }
    zones.append(zone)
    save_config()
    return jsonify({"ok": True, "zone": zone})


@app.route("/api/zones/<zone_id>", methods=["PUT"])
@login_required
def update_zone(zone_id):
    data = request.get_json(force=True)
    for z in zones:
        if z["id"] == zone_id:
            if "name" in data:
                z["name"] = data["name"]
            if "color" in data:
                z["color"] = normalize_zone_color(data.get("color"))
            if "zone_type" in data and data["zone_type"] in ("human", "vehicle_road"):
                z["zone_type"] = data["zone_type"]
            if "points" in data:
                z["points"] = data["points"]
            save_config()
            return jsonify({"ok": True, "zone": z})
    return jsonify({"ok": False, "error": "not found"}), 404


@app.route("/api/zones/<zone_id>", methods=["DELETE"])
@login_required
def delete_zone(zone_id):
    global zones
    zones = [z for z in zones if z["id"] != zone_id]
    motion_state.pop(zone_id, None)
    save_config()
    return jsonify({"ok": True})


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
