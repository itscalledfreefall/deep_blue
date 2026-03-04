#!/usr/bin/env python3
"""
Deep Blue Web - Forklift Safety System Dashboard
Event-driven state machine: camera capture, YOLO detection, MJPEG streaming,
ROI zone drawing, GPIO relay control, watchdog heartbeat, and web login.
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
MODEL_PATH = os.getenv("MODEL_PATH", "/home/enigma/yolo26n_ncnn_fp16")
FALLBACK_MODEL_PATH = "/home/enigma/yolo26n_256.onnx"
IMGSZ = 256
PERFORMANCE_MODE = True
CONF_THRESHOLD = 0.35 if PERFORMANCE_MODE else 0.20
CAM_WIDTH = 640
CAM_HEIGHT = 480
try:
    DIGITAL_ZOOM = float(os.getenv("DIGITAL_ZOOM", "1.0"))
except ValueError:
    DIGITAL_ZOOM = 1.0
if DIGITAL_ZOOM < 1.0:
    DIGITAL_ZOOM = 1.0
CAMERA_BACKEND = os.getenv("CAMERA_BACKEND", "ip").lower()
try:
    USB_CAMERA_INDEX = int(os.getenv("USB_CAMERA_INDEX", "0"))
except ValueError:
    USB_CAMERA_INDEX = 0
USB_CAPTURE_FOURCC = os.getenv("USB_CAPTURE_FOURCC", "YUYV").upper()
# IP camera (Ethernet RTSP/HTTP stream)
# Sub stream (640x480) for YOLO detection
IP_CAMERA_URL = os.getenv(
    "IP_CAMERA_URL",
    "rtsp://admin:matrix18@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"
).strip()
# Main stream (1280x720) for live view
IP_CAMERA_URL_MAIN = os.getenv(
    "IP_CAMERA_URL_MAIN",
    "rtsp://admin:matrix18@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
).strip()
IP_CAMERA_RECONNECT_DELAY = 2.0
PICAMERA2_USE_NOIR_TUNING = os.getenv("PICAMERA2_USE_NOIR_TUNING", "1").lower() not in {
    "0", "false", "no", "off"
}
PICAMERA2_TUNING_FILE = os.getenv("PICAMERA2_TUNING_FILE", "").strip()
PICAMERA2_NOIR_TUNING_CANDIDATES = [
    "/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json",
    "/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json",
    "/usr/share/libcamera/ipa/raspberrypi/imx219_noir.json",
]
CAMERA_READ_SLEEP = 0.005 if PERFORMANCE_MODE else 0.01
CAMERA_RETRY_DELAY = 1.0
CAMERA_MAX_READ_FAILURES = 30
CAMERA_DAY_CONTROLS = {
    "AeEnable": True,
    "AwbEnable": True,
    "AwbMode": 0,
    "Contrast": 1.10,
    "Saturation": 0.75
}
RGB_COLOR_GAINS = (1.0, 1.0, 1.0) if PERFORMANCE_MODE else (0.78, 1.00, 1.08)
DRAW_ZONE_FILL = os.getenv("DRAW_ZONE_FILL", "1").lower() not in {"0", "false", "no", "off"}
DRAW_OUTSIDE_BOXES = not PERFORMANCE_MODE
DRAW_BOX_LABELS = not PERFORMANCE_MODE
MJPEG_QUALITY = 72 if PERFORMANCE_MODE else 70
MJPEG_STREAM_SLEEP = 0.03 if PERFORMANCE_MODE else 0.05
PERSON_CLASS = 0
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TRAFFIC_RELAY_PIN = 23
WATCHDOG_RELAY_PIN = 24
# Camera-motion detection for vehicle_road zones
MOTION_BG_HISTORY = 300
MOTION_BG_VAR_THRESHOLD = 40
MOTION_WARMUP_FRAMES = 30

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
SECRET_KEY = "deepblue-forklift-safety-2026"
USERNAME = "deepblue"
PASSWORD = "matrix18"
DEFAULT_ZONE_COLOR = "#00e5ff"
HEX_COLOR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")

app = Flask(__name__)
app.secret_key = SECRET_KEY

# --- State Machine States ---
S_IDLE_SAFE = "IDLE_SAFE"
S_ROAD_MOTION_TRIGGERED = "ROAD_MOTION_TRIGGERED"
S_CLEARANCE_WAIT = "CLEARANCE_WAIT"
S_PEDESTRIAN_HOLD = "PEDESTRIAN_HOLD"
S_TRAFFIC_PASSAGE_GRANTED = "TRAFFIC_PASSAGE_GRANTED"
S_RECOVERY_COOLDOWN = "RECOVERY_COOLDOWN"
S_FAULT_SAFE = "FAULT_SAFE"

# --- Default Control Config ---
DEFAULT_CONTROL_CONFIG = {
    "clearance_seconds": 3.0,
    "pedestrian_clear_consecutive_frames": 2,
    "passage_seconds": 10.0,
    "cooldown_seconds": 2.0,
    "motion_consecutive_frames": 2,
    "motion_pixel_threshold": 0.04,
    "camera_failure_retry_limit": 10,
    "yolo_poll_interval_ms": 300,
    "watchdog_pulse_interval_ms": 1000,
    "watchdog_pulse_width_ms": 100,
}

CONTROL_CONFIG_BOUNDS = {
    "clearance_seconds":                 (0.0, 30.0),
    "pedestrian_clear_consecutive_frames": (1, 30),
    "passage_seconds":                   (1.0, 120.0),
    "cooldown_seconds":                  (0.5, 30.0),
    "motion_consecutive_frames":         (1, 30),
    "motion_pixel_threshold":            (0.005, 0.5),
    "camera_failure_retry_limit":        (1, 100),
    "yolo_poll_interval_ms":             (50, 5000),
    "watchdog_pulse_interval_ms":        (100, 10000),
    "watchdog_pulse_width_ms":           (10, 5000),
}

# --- Shared State ---
lock = threading.Lock()
latest_frame = None        # sub stream (640x480) for YOLO
latest_frame_hd = None     # main stream (HD) for live view, None if single-stream
annotated_frame = None
zones = []
MAX_ZONES = 8
control_config = dict(DEFAULT_CONTROL_CONFIG)

# Controller state (read by API, written by detection thread)
controller_state = {
    "state": S_IDLE_SAFE,
    "state_since_ts": 0.0,
    "trigger_zone_id": None,
    "fault_reason": None,
    "event_id": 0,
    "ped_clear_count": 0,
}

status = {
    "fps": 0.0,
    "persons": 0,
    "vehicles": 0,
    "motion_zones": 0,
    "relay": "OFF",
    "running": False,
    "control_state": S_IDLE_SAFE,
    "state_since_ts": 0.0,
    "pending_trigger_zone_id": None,
    "pedestrian_hold": False,
    "fault_active": False,
    "fault_reason": None,
    "watchdog_ok": True,
    "event_mode": "event_driven",
    "yolo_mode": "sleeping",
}

# Camera-motion detection shared state
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=MOTION_BG_HISTORY,
    varThreshold=MOTION_BG_VAR_THRESHOLD,
    detectShadows=False
)
motion_state = {}
motion_frame_count = 0
camera_reconnected = False

# Watchdog state
watchdog_last_pulse_ts = 0.0
watchdog_ok = True


def load_config():
    global zones, control_config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            if "zones" in data:
                zones = data["zones"]
                for z in zones:
                    z["color"] = normalize_zone_color(z.get("color", DEFAULT_ZONE_COLOR))
                    if "zone_type" not in z:
                        z["zone_type"] = "human"
            elif "roi" in data and data["roi"]:
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
            if "control_config" in data and isinstance(data["control_config"], dict):
                merged = dict(DEFAULT_CONTROL_CONFIG)
                merged.update(data["control_config"])
                control_config = merged
        except Exception:
            zones = []


def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump({"zones": zones, "control_config": control_config}, f)


def validate_control_config(incoming):
    errors = []
    validated = {}
    for key, (lo, hi) in CONTROL_CONFIG_BOUNDS.items():
        if key not in incoming:
            continue
        val = incoming[key]
        if isinstance(lo, int) and isinstance(hi, int):
            if not isinstance(val, (int, float)):
                errors.append(f"{key}: must be numeric")
                continue
            val = int(val)
        else:
            if not isinstance(val, (int, float)):
                errors.append(f"{key}: must be numeric")
                continue
            val = float(val)
        if val < lo or val > hi:
            errors.append(f"{key}: must be between {lo} and {hi}")
            continue
        validated[key] = val
    return validated, errors


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
traffic_relay_active = False


def setup_gpio():
    if not GPIO_AVAILABLE:
        print("WARNING: RPi.GPIO not available, relays disabled")
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(TRAFFIC_RELAY_PIN, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(WATCHDOG_RELAY_PIN, GPIO.OUT, initial=GPIO.HIGH)
    print(f"Traffic relay: BCM {TRAFFIC_RELAY_PIN} (active-low)")
    print(f"Watchdog relay: BCM {WATCHDOG_RELAY_PIN} (active-low)")


def set_traffic_relay(active):
    global traffic_relay_active
    if traffic_relay_active == active:
        return
    traffic_relay_active = active
    status["relay"] = "ON" if active else "OFF"
    if GPIO_AVAILABLE:
        state = GPIO.LOW if active else GPIO.HIGH
        GPIO.output(TRAFFIC_RELAY_PIN, state)


def watchdog_pulse():
    if not GPIO_AVAILABLE:
        return
    width_s = control_config["watchdog_pulse_width_ms"] / 1000.0
    GPIO.output(WATCHDOG_RELAY_PIN, GPIO.LOW)
    time.sleep(width_s)
    GPIO.output(WATCHDOG_RELAY_PIN, GPIO.HIGH)


def watchdog_thread_fn():
    global watchdog_last_pulse_ts, watchdog_ok
    while status["running"]:
        interval_s = control_config["watchdog_pulse_interval_ms"] / 1000.0
        try:
            watchdog_pulse()
            watchdog_last_pulse_ts = time.time()
            watchdog_ok = True
            status["watchdog_ok"] = True
        except Exception as e:
            print(f"WARNING: watchdog pulse failed: {e}")
            watchdog_ok = False
            status["watchdog_ok"] = False
        time.sleep(interval_s)
    print("Watchdog thread stopped")


# --- Camera helpers ---
def apply_camera_controls(picam2, controls):
    for key, value in controls.items():
        try:
            picam2.set_controls({key: value})
        except Exception:
            pass


def apply_rgb_color_gains(frame, gains):
    if gains == (1.0, 1.0, 1.0):
        return frame
    fr = frame.astype(np.float32)
    fr[..., 0] *= gains[0]
    fr[..., 1] *= gains[1]
    fr[..., 2] *= gains[2]
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


def detect_motion_in_zone(fg_mask, zone_points, frame_shape, threshold):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(zone_points, dtype=np.int32)], 255)
    zone_fg = cv2.bitwise_and(fg_mask, mask)
    zone_area = cv2.countNonZero(mask)
    if zone_area == 0:
        return False, 0.0
    ratio = cv2.countNonZero(zone_fg) / zone_area
    return ratio >= threshold, ratio


def create_picamera2_instance():
    if not PICAMERA2_AVAILABLE:
        return None
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
            return path
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


# --- Camera Threads ---
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
            frame = picam2.capture_array().copy()
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


def _ip_hd_reader():
    """Background reader for the main/HD IP camera stream (live view only)."""
    global latest_frame_hd
    url = IP_CAMERA_URL_MAIN
    if not url:
        return
    print(f"Connecting to IP camera HD stream: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"WARNING: cannot open HD stream, live view will use sub stream")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"IP camera HD stream started: {actual_w}x{actual_h}")
    read_failures = 0
    while status["running"]:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            read_failures += 1
            if read_failures >= CAMERA_MAX_READ_FAILURES:
                print("WARNING: HD stream lost, live view falls back to sub stream")
                break
            time.sleep(0.02)
            continue
        read_failures = 0
        with lock:
            latest_frame_hd = bgr  # keep as BGR, detection thread will annotate
        time.sleep(CAMERA_READ_SLEEP)
    cap.release()
    with lock:
        latest_frame_hd = None
    print("IP camera HD stream stopped")


def camera_thread_ip():
    global latest_frame, camera_reconnected
    url = IP_CAMERA_URL
    if not url:
        print("ERROR: IP_CAMERA_URL not set")
        with lock:
            latest_frame = None
        return False
    # Start HD stream reader in background
    hd_thread = threading.Thread(target=_ip_hd_reader, daemon=True)
    hd_thread.start()
    print(f"Connecting to IP camera sub stream: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"ERROR: cannot open IP camera stream: {url}")
        with lock:
            latest_frame = None
        return False
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"IP camera sub stream started: {actual_w}x{actual_h}")
    camera_reconnected = True
    restart_required = False
    read_failures = 0
    while status["running"]:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            read_failures += 1
            if read_failures >= CAMERA_MAX_READ_FAILURES:
                print("WARNING: IP camera read failed repeatedly, reconnecting")
                restart_required = True
                break
            time.sleep(0.02)
            continue
        read_failures = 0
        # Resize to standard dimensions if needed
        h, w = bgr.shape[:2]
        if w != CAM_WIDTH or h != CAM_HEIGHT:
            bgr = cv2.resize(bgr, (CAM_WIDTH, CAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
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
    print("IP camera stopped")
    return True


def camera_thread():
    backend = CAMERA_BACKEND
    if backend not in {"auto", "usb", "picamera2", "ip"}:
        print(f"WARNING: invalid CAMERA_BACKEND={backend}, using auto")
        backend = "auto"
    print(f"Camera backend mode: {backend}")
    while status["running"]:
        if backend == "ip":
            if camera_thread_ip():
                return
            print(f"WARNING: retrying IP camera in {IP_CAMERA_RECONNECT_DELAY:.1f}s")
            time.sleep(IP_CAMERA_RECONNECT_DELAY)
            continue
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


# --- State Machine Helpers ---
def transition_state(new_state, reason=None, trigger_zone=None):
    now = time.time()
    old = controller_state["state"]
    controller_state["state"] = new_state
    controller_state["state_since_ts"] = now
    if trigger_zone is not None:
        controller_state["trigger_zone_id"] = trigger_zone
    if new_state == S_FAULT_SAFE:
        controller_state["fault_reason"] = reason
    elif old == S_FAULT_SAFE:
        controller_state["fault_reason"] = None
    if new_state == S_IDLE_SAFE:
        controller_state["trigger_zone_id"] = None
        controller_state["ped_clear_count"] = 0
    sync_status()


def enter_fault(reason):
    print(f"FAULT: {reason}")
    set_traffic_relay(False)
    transition_state(S_FAULT_SAFE, reason=reason)


def sync_status():
    cs = controller_state
    status["control_state"] = cs["state"]
    status["state_since_ts"] = cs["state_since_ts"]
    status["pending_trigger_zone_id"] = cs["trigger_zone_id"]
    status["pedestrian_hold"] = cs["state"] == S_PEDESTRIAN_HOLD
    status["fault_active"] = cs["state"] == S_FAULT_SAFE
    status["fault_reason"] = cs["fault_reason"]


def classify_zones(current_zones):
    human_zones = []
    road_zones = []
    for z in current_zones:
        if z.get("zone_type", "human") == "vehicle_road":
            road_zones.append(z)
        else:
            human_zones.append(z)
    return human_zones, road_zones


def check_zone_health(human_zones, road_zones):
    if not road_zones:
        return "no_vehicle_road_zone", "warning"
    if not human_zones:
        return "no_human_zone", "warning"
    return None, None


# --- Detection Thread (State Machine) ---
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
        enter_fault("model_load_failed")
        while status["running"]:
            time.sleep(0.5)
        return
    print("YOLO warmup done")

    # Initialize state
    transition_state(S_IDLE_SAFE)
    set_traffic_relay(False)

    consecutive_frame_failures = 0
    consecutive_yolo_failures = 0
    event_counter = 0

    while status["running"]:
        cfg = dict(control_config)  # snapshot for this cycle

        # Read frame
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            consecutive_frame_failures += 1
            if consecutive_frame_failures >= cfg["camera_failure_retry_limit"]:
                if controller_state["state"] != S_FAULT_SAFE:
                    enter_fault("camera_frame_loss")
            time.sleep(0.05)
            continue
        consecutive_frame_failures = 0

        # If in FAULT_SAFE due to camera, recover
        if (controller_state["state"] == S_FAULT_SAFE
                and controller_state["fault_reason"] == "camera_frame_loss"):
            transition_state(S_IDLE_SAFE)

        t0 = time.time()
        now = t0
        current_zones = [z for z in zones if len(z.get("points", [])) >= 3]
        human_zones, road_zones = classify_zones(current_zones)

        # Zone health check (warning only, no fault)
        zone_issue, _ = check_zone_health(human_zones, road_zones)
        status["zone_warning"] = zone_issue

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

        # --- Motion detection (runs every cycle) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        fg_mask = bg_subtractor.apply(gray)
        motion_frame_count += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        motion_warmup_done = motion_frame_count > MOTION_WARMUP_FRAMES

        motion_zones_active = 0
        active_motion_zone_ids = set()
        triggering_zone_id = None
        zone_pixel_pcts = {}  # zone_id -> live pixel change %
        if road_zones and motion_warmup_done:
            for zone in road_zones:
                zid = zone["id"]
                has_motion, pixel_ratio = detect_motion_in_zone(
                    fg_mask, zone["points"], frame.shape,
                    cfg["motion_pixel_threshold"]
                )
                zone_pixel_pcts[zid] = round(pixel_ratio * 100, 2)
                if zid not in motion_state:
                    motion_state[zid] = {"consecutive_frames": 0, "last_detected_ts": 0.0}
                if has_motion:
                    motion_state[zid]["consecutive_frames"] += 1
                    if motion_state[zid]["consecutive_frames"] >= cfg["motion_consecutive_frames"]:
                        motion_state[zid]["last_detected_ts"] = now
                        motion_zones_active += 1
                        active_motion_zone_ids.add(zid)
                        if triggering_zone_id is None:
                            triggering_zone_id = zid
                else:
                    motion_state[zid]["consecutive_frames"] = 0

        # Clean stale motion_state entries
        active_zone_ids = {z["id"] for z in current_zones}
        for zid in list(motion_state):
            if zid not in active_zone_ids:
                del motion_state[zid]

        # --- YOLO inference (conditional on state) ---
        cs = controller_state["state"]
        run_yolo = cs in (S_PEDESTRIAN_HOLD, S_CLEARANCE_WAIT)
        # Also run in IDLE_SAFE if no road zones (legacy fallback: full-frame detection)
        if cs == S_IDLE_SAFE and not road_zones and human_zones:
            run_yolo = True

        persons_in_roi = []
        persons_outside = []
        vehicles_in_roi = []
        vehicles_outside = []

        if run_yolo:
            status["yolo_mode"] = "active"
            try:
                results = model(frame, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)
                consecutive_yolo_failures = 0
            except Exception as e:
                print(f"WARNING: inference failed, reloading model: {e}")
                consecutive_yolo_failures += 1
                if consecutive_yolo_failures >= 3:
                    enter_fault("yolo_inference_failed")
                    # Try to reload for next cycle
                    model = load_model_with_warmup(frame)
                    if model is None:
                        enter_fault("model_load_failed")
                        while status["running"]:
                            time.sleep(0.5)
                        return
                    consecutive_yolo_failures = 0
                else:
                    model = load_model_with_warmup(frame)
                    if model is None:
                        enter_fault("model_load_failed")
                        while status["running"]:
                            time.sleep(0.5)
                        return
                results = None

            if results is not None:
                all_persons = []
                all_vehicles = []
                for b in results[0].boxes:
                    cls_id = int(b.cls)
                    if cls_id == PERSON_CLASS:
                        all_persons.append(b)
                    elif cls_id in VEHICLE_CLASSES:
                        all_vehicles.append(b)

                persons_outside = list(all_persons)
                vehicles_outside = list(all_vehicles)
                if human_zones:
                    for zone in human_zones:
                        roi_contour = np.array(zone["points"], dtype=np.int32)
                        p_in, persons_outside = filter_by_roi(persons_outside, roi_contour)
                        persons_in_roi.extend(p_in)
                        v_in, vehicles_outside = filter_by_roi(vehicles_outside, roi_contour)
                        vehicles_in_roi.extend(v_in)
                elif not current_zones:
                    persons_in_roi, persons_outside = all_persons, []
                    vehicles_in_roi, vehicles_outside = all_vehicles, []
        else:
            status["yolo_mode"] = "sleeping"

        pedestrian_present = len(persons_in_roi) > 0

        # --- State machine transitions ---
        # Re-read current state (fault handler may have changed it mid-cycle)
        cs = controller_state["state"]
        elapsed_in_state = now - controller_state["state_since_ts"]

        if cs == S_IDLE_SAFE:
            set_traffic_relay(False)
            if motion_zones_active > 0:
                event_counter += 1
                controller_state["event_id"] = event_counter
                transition_state(S_ROAD_MOTION_TRIGGERED, trigger_zone=triggering_zone_id)
            elif not road_zones and human_zones and len(vehicles_in_roi) > 0 and not pedestrian_present:
                # Legacy fallback: no road zones, vehicle detected in human zone
                event_counter += 1
                controller_state["event_id"] = event_counter
                transition_state(S_ROAD_MOTION_TRIGGERED)

        elif cs == S_ROAD_MOTION_TRIGGERED:
            set_traffic_relay(False)
            transition_state(S_CLEARANCE_WAIT)

        elif cs == S_CLEARANCE_WAIT:
            set_traffic_relay(False)
            if elapsed_in_state >= cfg["clearance_seconds"]:
                transition_state(S_PEDESTRIAN_HOLD)

        elif cs == S_PEDESTRIAN_HOLD:
            set_traffic_relay(False)
            if pedestrian_present:
                controller_state["ped_clear_count"] = 0
            else:
                controller_state["ped_clear_count"] += 1
                if controller_state["ped_clear_count"] >= cfg["pedestrian_clear_consecutive_frames"]:
                    transition_state(S_TRAFFIC_PASSAGE_GRANTED)
                    set_traffic_relay(True)

        elif cs == S_TRAFFIC_PASSAGE_GRANTED:
            # Relay stays on for configured duration
            if elapsed_in_state >= cfg["passage_seconds"]:
                set_traffic_relay(False)
                transition_state(S_RECOVERY_COOLDOWN)

        elif cs == S_RECOVERY_COOLDOWN:
            set_traffic_relay(False)
            if elapsed_in_state >= cfg["cooldown_seconds"]:
                transition_state(S_IDLE_SAFE)

        elif cs == S_FAULT_SAFE:
            set_traffic_relay(False)
            # Auto-recover from recoverable faults
            if controller_state["fault_reason"] == "camera_frame_loss":
                pass  # recovered earlier when frames return
            elif controller_state["fault_reason"] == "yolo_inference_failed":
                transition_state(S_IDLE_SAFE)

        sync_status()

        # --- FPS ---
        elapsed = time.time() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0

        # --- Annotate frame ---
        # Use HD frame for live view if available, otherwise sub stream
        with lock:
            hd_bgr = latest_frame_hd.copy() if latest_frame_hd is not None else None
        if hd_bgr is not None:
            display = hd_bgr
            dh, dw = display.shape[:2]
            sx = dw / CAM_WIDTH
            sy = dh / CAM_HEIGHT
        else:
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            dh, dw = CAM_HEIGHT, CAM_WIDTH
            sx, sy = 1.0, 1.0

        def scale_pts(pts):
            if sx == 1.0 and sy == 1.0:
                return pts
            return (pts.astype(np.float64) * np.array([sx, sy])).astype(np.int32)

        def scale_box(x1, y1, x2, y2):
            return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)

        line_w = max(2, int(2 * min(sx, sy)))
        font_scale = 0.5 * min(sx, sy)

        if current_zones:
            if DRAW_ZONE_FILL:
                overlay = display.copy()
                for zone in current_zones:
                    pts = scale_pts(np.array(zone["points"], dtype=np.int32))
                    bgr = hex_to_bgr(zone.get("color", "#00e5ff"))
                    alpha = 0.2 if zone["id"] in active_motion_zone_ids else 0.08
                    cv2.fillPoly(overlay, [pts], bgr)
                    cv2.addWeighted(overlay, alpha, display, 1.0 - alpha, 0, display)
                    overlay = display.copy()
            for zone in current_zones:
                pts = scale_pts(np.array(zone["points"], dtype=np.int32))
                bgr = hex_to_bgr(zone.get("color", "#00e5ff"))
                cv2.polylines(display, [pts], True, bgr, line_w)
                name = zone.get("name", "")
                if DRAW_BOX_LABELS and name:
                    (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                    cx = int(np.mean(pts[:, 0])) - tw // 2
                    cy = int(np.mean(pts[:, 1])) + th // 2
                    cv2.putText(display, name, (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4)
                    cv2.putText(display, name, (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        for box in persons_in_roi:
            x1, y1, x2, y2 = scale_box(*map(int, box.xyxy[0]))
            conf = float(box.conf)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), line_w)
            if DRAW_BOX_LABELS:
                cv2.putText(display, f"PERSON {conf:.0%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        if DRAW_OUTSIDE_BOXES:
            for box in persons_outside:
                x1, y1, x2, y2 = scale_box(*map(int, box.xyxy[0]))
                cv2.rectangle(display, (x1, y1), (x2, y2), (128, 128, 128), 1)

        for box in vehicles_in_roi:
            x1, y1, x2, y2 = scale_box(*map(int, box.xyxy[0]))
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = VEHICLE_CLASSES.get(cls_id, "vehicle").upper()
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 255), line_w)
            if DRAW_BOX_LABELS:
                cv2.putText(display, f"{label} {conf:.0%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 220, 255), 2)
        if DRAW_OUTSIDE_BOXES:
            for box in vehicles_outside:
                x1, y1, x2, y2 = scale_box(*map(int, box.xyxy[0]))
                cv2.rectangle(display, (x1, y1), (x2, y2), (128, 128, 128), 1)

        # Status bar
        state_short = controller_state["state"].replace("_", " ")
        relay_color = (0, 200, 0) if traffic_relay_active else (0, 0, 255)
        relay_text = "PASS" if traffic_relay_active else "SAFE"
        status_text = f"{state_short} | {relay_text} | FPS:{fps:.1f}"
        if motion_zones_active > 0:
            status_text += f" M:{motion_zones_active}"
        if len(persons_in_roi) > 0:
            status_text += f" P:{len(persons_in_roi)}"
        cv2.putText(display, status_text,
                    (10, dh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * min(sx, sy), relay_color, line_w)

        with lock:
            annotated_frame = display

        status["fps"] = round(fps, 1)
        status["persons"] = len(persons_in_roi)
        status["vehicles"] = len(vehicles_in_roi)
        status["motion_zones"] = motion_zones_active
        status["zone_pixel_pcts"] = zone_pixel_pcts

        # Throttle YOLO polling when in pedestrian hold
        if controller_state["state"] == S_PEDESTRIAN_HOLD:
            yolo_sleep = cfg["yolo_poll_interval_ms"] / 1000.0
            if yolo_sleep > 0:
                time.sleep(yolo_sleep)

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


@app.route("/api/control-config", methods=["GET"])
@login_required
def get_control_config():
    return jsonify(control_config)


@app.route("/api/control-config", methods=["PUT"])
@login_required
def put_control_config():
    global control_config
    data = request.get_json(force=True)
    validated, errors = validate_control_config(data)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    if not validated:
        return jsonify({"ok": False, "errors": ["no valid fields provided"]}), 400
    control_config.update(validated)
    save_config()
    return jsonify({"ok": True, "control_config": control_config})


# --- Main ---
def shutdown_handler(signum, frame):
    status["running"] = False
    if GPIO_AVAILABLE:
        GPIO.output(TRAFFIC_RELAY_PIN, GPIO.HIGH)
        GPIO.output(WATCHDOG_RELAY_PIN, GPIO.HIGH)
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
    wd_t = threading.Thread(target=watchdog_thread_fn, daemon=True)
    cam_t.start()
    det_t.start()
    wd_t.start()

    print("Starting Deep Blue Web on http://0.0.0.0:80")
    app.run(host="0.0.0.0", port=80, threaded=True, debug=False)
