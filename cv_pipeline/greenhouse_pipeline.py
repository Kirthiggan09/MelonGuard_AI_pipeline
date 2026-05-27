#!/usr/bin/env python3
"""
greenhouse_pipeline.py
======================
Greenhouse Precision Agriculture — Rail Camera Computer Vision Pipeline

Production-grade real-time rockmelon leaf disease detection system for
hardwired rail-mounted cameras inside greenhouse structures.

Modules:
    1. VideoStreamCapture  — RTSP / local camera ingestion with auto-reconnect
    2. InferenceEngine     — YOLO model inference with confidence filtering
    3. Visualizer          — Anti-aliased bounding box overlay & HUD
    4. AlertEngine         — Structured JSON alert logging & temporal tracking

Usage:
    # Live webcam (for testing)
    python greenhouse_pipeline.py --source 0

    # RTSP rail camera
    python greenhouse_pipeline.py --source rtsp://192.168.1.100:554/stream

    # Offline: run on a folder of images
    python greenhouse_pipeline.py --source ../path/to/images --mode images

    # Offline: run on a single image
    python greenhouse_pipeline.py --source leaf_photo.jpg --mode images

Controls (live/video modes):
    q / ESC     Quit
    s           Save screenshot
    +/-         Adjust confidence threshold
    t           Toggle HUD overlay
    p           Pause / resume
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys
import json
import time
import logging
import argparse
import threading
import queue
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "best.pt"
OUTPUT_DIR = SCRIPT_DIR / "detection_results"
ALERT_DIR  = SCRIPT_DIR / "alerts"

# ── Inference defaults ──────────────────────────────────────────────────────
CONF_THRESHOLD   = 0.75      # §5: 75% minimum confidence to register a detection
IMG_SIZE         = 640       # YOLO inference resolution
INTERNAL_CONF    = 0.10      # low internal threshold; we filter at display/alert level

# ── Video capture defaults ──────────────────────────────────────────────────
DEFAULT_SOURCE      = "0"                  # webcam index or RTSP URL
TARGET_WIDTH        = 1920                 # request 1080p
TARGET_HEIGHT       = 1080
RECONNECT_ATTEMPTS  = 10
RECONNECT_BASE_SEC  = 3.0                 # exponential backoff base

# ── Alert risk classes (any non-normal triggers an alert) ───────────────────
RISK_CLASSES = {"aphid", "fungus", "leaf miner", "unknown"}

# ── Class display metadata (§4: exact spec definitions) ─────────────────────
CLASS_DISPLAY = {
    # model_index: (display_name, short_symptom_description)
    0: ("Aphid",      "Severe inward curling / distortion due to pest infestation"),
    1: ("Fungus",     "Bluish flour-like dust + opposite yellow dots (sporangia)"),
    2: ("Leaf Miner", "Light green to white squiggly trails / feeding tunnels"),
    3: ("Normal",     "Healthy green leaves — no chlorosis or structural curling"),
    4: ("Unknown",    "Anomalous features from chemical / pesticide spray residue"),
}

# ── Colour palette per class (BGR) ──────────────────────────────────────────
CLASS_COLORS = {
    0: (0,  100, 255),   # Aphid       → orange
    1: (50,  50, 220),   # Fungus      → red
    2: (0,  200, 200),   # Leaf Miner  → yellow
    3: (0,  200,   0),   # Normal      → green
    4: (200, 150,  0),   # Unknown     → teal
}
DEFAULT_COLOR = (180, 180, 180)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("greenhouse")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Detection:
    """Single bounding-box detection result."""
    class_id:     int
    class_name:   str
    display_name: str
    confidence:   float          # 0.0 – 1.0
    bbox:         Tuple[int, int, int, int]   # (x1, y1, x2, y2) pixel coords
    symptom:      str = ""

    @property
    def label(self) -> str:
        return f"{self.display_name}: {self.confidence * 100:.1f}%"

    @property
    def is_risk(self) -> bool:
        return self.class_name in RISK_CLASSES


@dataclass
class AlertPayload:
    """Structured JSON alert record (§5)."""
    timestamp:             str
    rail_position_id:      int
    detected_disease:      str
    confidence_percentage: float
    bbox:                  Tuple[int, int, int, int]
    frame_id:              int
    symptom_description:   str = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — VIDEO STREAMING CAPTURE
# ═══════════════════════════════════════════════════════════════════════════════
class VideoStreamCapture:
    """
    Threaded video capture with RTSP reconnection and local camera fallback.

    Continuously reads frames in a background thread so the main inference
    loop is never blocked by I/O latency or network hiccups.
    """

    def __init__(self, source: str, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT):
        self.source = source
        self.width  = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0

        self._is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp")

    # ── Public API ──────────────────────────────────────────────────────────
    def start(self) -> bool:
        """Open the video source and start the reader thread."""
        ok = self._open_source()
        if not ok:
            return False
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the most recent frame (thread-safe, non-blocking)."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        """Stop the reader thread and release the capture device."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._release()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (0, 0)

    # ── Internal ────────────────────────────────────────────────────────────
    def _open_source(self) -> bool:
        """Open the capture device, requesting the target resolution."""
        self._release()
        src = self.source
        # Determine if source is an integer camera index
        try:
            src = int(src)
        except (ValueError, TypeError):
            pass

        log.info(f"Opening video source: {src}")
        self._cap = cv2.VideoCapture(src)
        time.sleep(0.3)

        if not self._cap.isOpened():
            log.error(f"Failed to open video source: {src}")
            return False

        # Request resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        actual_w, actual_h = self.resolution
        log.info(f"Video source opened — resolution: {actual_w}x{actual_h}")
        return True

    def _release(self):
        try:
            if self._cap and self._cap.isOpened():
                self._cap.release()
        except Exception:
            pass
        self._cap = None

    def _reader_loop(self):
        """Background thread: continuously grab the latest frame."""
        consecutive_failures = 0
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if not self._try_reconnect():
                    break
                consecutive_failures = 0
                continue

            ret, frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    log.warning("Too many consecutive read failures — attempting reconnect")
                    if not self._try_reconnect():
                        break
                    consecutive_failures = 0
                time.sleep(0.01)
                continue

            consecutive_failures = 0
            self._frame_count += 1
            with self._lock:
                self._frame = frame

        log.info("Video reader thread stopped.")

    def _try_reconnect(self) -> bool:
        """Attempt reconnection with exponential backoff (RTSP only)."""
        if not self._is_rtsp:
            log.error("Local camera disconnected — cannot auto-reconnect.")
            return False

        for attempt in range(1, RECONNECT_ATTEMPTS + 1):
            delay = RECONNECT_BASE_SEC * (1.5 ** (attempt - 1))
            log.warning(f"RTSP reconnect attempt {attempt}/{RECONNECT_ATTEMPTS} in {delay:.1f}s...")
            time.sleep(delay)
            if self._open_source():
                log.info("Reconnected successfully.")
                return True

        log.error("All reconnection attempts exhausted.")
        self._running = False
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — MODEL INFERENCE & DECODING
# ═══════════════════════════════════════════════════════════════════════════════
class InferenceEngine:
    """
    Loads the YOLO model and runs inference, decoding raw predictions into
    structured `Detection` objects filtered by the configurable confidence
    threshold.
    """

    def __init__(self, model_path: str = str(MODEL_PATH), conf_threshold: float = CONF_THRESHOLD):
        self.conf_threshold = conf_threshold
        log.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self._class_names: Dict[int, str] = self.model.names
        log.info(f"Model classes (native): {self._class_names}")
        log.info(f"Model task: {self.model.task}")
        log.info(f"Confidence threshold: {self.conf_threshold * 100:.0f}%")

    def infer(self, frame: np.ndarray) -> Tuple[List[Detection], object]:
        """
        Run inference on a single BGR frame.

        Returns:
            detections: List of Detection objects that pass the confidence threshold.
            raw_results: The raw Ultralytics Results object (for advanced use).
        """
        # Run YOLO with a low internal threshold — we filter ourselves
        raw_results = self.model(frame, conf=INTERNAL_CONF, imgsz=IMG_SIZE, verbose=False)

        detections: List[Detection] = []
        boxes = raw_results[0].boxes

        for box in boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            cls_id = int(box.cls[0])
            cls_name = self._class_names.get(cls_id, f"class_{cls_id}")
            display_name, symptom = CLASS_DISPLAY.get(cls_id, (cls_name, ""))

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            detections.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                display_name=display_name,
                confidence=conf,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                symptom=symptom,
            ))

        return detections, raw_results

    def decode_labels(self, detections: List[Detection]) -> List[str]:
        """
        §5 output format: ["Leaf Miner: 94.2%", "Fungus: 78.1%"]
        """
        return [d.label for d in detections]

    @property
    def class_names(self) -> Dict[int, str]:
        return self._class_names


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — VISUALIZATION & ANNOTATION OVERLAY
# ═══════════════════════════════════════════════════════════════════════════════
class Visualizer:
    """
    Draws anti-aliased bounding boxes, label banners, a colour legend,
    and a heads-up display bar onto inference frames.
    """

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.60
    THICKNESS  = 2

    def __init__(self, class_names: Dict[int, str], show_hud: bool = True):
        self.class_names = class_names
        self.show_hud = show_hud

    # ── Public API ──────────────────────────────────────────────────────────
    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float = 0.0,
        conf_threshold: float = CONF_THRESHOLD,
        rail_position: int = 0,
        paused: bool = False,
    ) -> np.ndarray:
        """Return a fully annotated copy of the frame."""
        canvas = frame.copy()

        # Draw bounding boxes & labels
        for det in detections:
            self._draw_bbox(canvas, det)

        # Draw legend
        self._draw_legend(canvas)

        # Draw HUD
        if self.show_hud:
            self._draw_hud(canvas, fps, conf_threshold, len(detections), rail_position, paused)

        return canvas

    # ── Bounding box ────────────────────────────────────────────────────────
    def _draw_bbox(self, img: np.ndarray, det: Detection):
        x1, y1, x2, y2 = det.bbox
        color = CLASS_COLORS.get(det.class_id, DEFAULT_COLOR)

        # Anti-aliased rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Label banner
        label = det.label
        (tw, th), baseline = cv2.getTextSize(label, self.FONT, self.FONT_SCALE, self.THICKNESS)
        label_y = max(y1, th + 10)
        # Background pill
        cv2.rectangle(img, (x1, label_y - th - 10), (x1 + tw + 12, label_y + 4), color, -1, cv2.LINE_AA)
        # Text
        cv2.putText(img, label, (x1 + 6, label_y - 4), self.FONT, self.FONT_SCALE,
                    (255, 255, 255), self.THICKNESS, cv2.LINE_AA)

    # ── Colour legend ───────────────────────────────────────────────────────
    def _draw_legend(self, img: np.ndarray):
        pad, line_h = 10, 24
        legend_w = 200
        n = len(CLASS_DISPLAY)
        legend_h = pad * 2 + line_h * n
        x0 = img.shape[1] - legend_w - 12
        y0 = 12

        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + legend_w, y0 + legend_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        for i, (cid, (dname, _)) in enumerate(sorted(CLASS_DISPLAY.items())):
            cy = y0 + pad + i * line_h + 16
            color = CLASS_COLORS.get(cid, DEFAULT_COLOR)
            cv2.rectangle(img, (x0 + pad, cy - 12), (x0 + pad + 16, cy + 2), color, -1, cv2.LINE_AA)
            cv2.putText(img, dname, (x0 + pad + 22, cy), self.FONT, 0.48,
                        (240, 240, 240), 1, cv2.LINE_AA)

    # ── HUD bar ─────────────────────────────────────────────────────────────
    def _draw_hud(self, img: np.ndarray, fps: float, conf: float,
                  det_count: int, rail_pos: int, paused: bool):
        h, w = img.shape[:2]
        bar_h = 40

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        status = "⏸ PAUSED" if paused else "● LIVE"
        info = (
            f"{status}  |  FPS: {fps:.0f}  |  Conf: {conf*100:.0f}%  |  "
            f"Detections: {det_count}  |  Rail Pos: {rail_pos}  |  "
            f"[S]creenshot  [T]oggle HUD  [P]ause  [Q]uit"
        )
        cv2.putText(img, info, (12, 27), self.FONT, 0.45, (180, 255, 180), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 4 — ALERT TRIGGER FRAMEWORK & TEMPORAL TRACKING
# ═══════════════════════════════════════════════════════════════════════════════
class TrackingLedger:
    """
    Temporal tracking stub (§5).

    Records per-position detection history across consecutive runs.
    Architecture is ready for ByteTrack/SORT integration once the rail
    controller provides real position IDs.

    Structure:
        ledger[rail_position_id][date_str] = [
            {"class": ..., "confidence": ..., "timestamp": ...}, ...
        ]
    """

    def __init__(self, ledger_path: Optional[str] = None):
        self._ledger_path = ledger_path or str(ALERT_DIR / "tracking_ledger.json")
        self._ledger: Dict[int, Dict[str, list]] = {}
        self._load()

    def record(self, rail_position: int, class_name: str, confidence: float):
        date_key = datetime.now().strftime("%Y-%m-%d")
        pos_key = str(rail_position)

        if pos_key not in self._ledger:
            self._ledger[pos_key] = {}
        if date_key not in self._ledger[pos_key]:
            self._ledger[pos_key][date_key] = []

        self._ledger[pos_key][date_key].append({
            "class":      class_name,
            "confidence": round(confidence, 4),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        })

    def save(self):
        os.makedirs(os.path.dirname(self._ledger_path), exist_ok=True)
        with open(self._ledger_path, "w") as f:
            json.dump(self._ledger, f, indent=2)
        log.info(f"Tracking ledger saved: {self._ledger_path}")

    def get_history(self, rail_position: int) -> Dict[str, list]:
        return self._ledger.get(str(rail_position), {})

    def _load(self):
        if os.path.exists(self._ledger_path):
            try:
                with open(self._ledger_path, "r") as f:
                    self._ledger = json.load(f)
                log.info(f"Loaded existing tracking ledger ({len(self._ledger)} positions)")
            except Exception:
                self._ledger = {}


class AlertEngine:
    """
    Evaluates detections against risk criteria and emits structured JSON
    alert payloads to local `.jsonl` log files.
    """

    def __init__(self, alert_dir: str = str(ALERT_DIR)):
        self.alert_dir = Path(alert_dir)
        self.alert_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = TrackingLedger()
        self._alert_count = 0

    def evaluate(
        self,
        detections: List[Detection],
        rail_position: int,
        frame_id: int,
    ) -> List[AlertPayload]:
        """
        Check each detection against risk criteria.
        Returns list of alert payloads for detections that trigger.
        """
        alerts: List[AlertPayload] = []

        for det in detections:
            if not det.is_risk:
                continue

            payload = AlertPayload(
                timestamp=datetime.now(timezone.utc).isoformat(),
                rail_position_id=rail_position,
                detected_disease=det.display_name,
                confidence_percentage=round(det.confidence * 100, 2),
                bbox=det.bbox,
                frame_id=frame_id,
                symptom_description=det.symptom,
            )
            alerts.append(payload)

            # Record in temporal ledger
            self.ledger.record(rail_position, det.class_name, det.confidence)

        if alerts:
            self._write_alerts(alerts)

        return alerts

    @property
    def alert_count(self) -> int:
        return self._alert_count

    def shutdown(self):
        """Persist the tracking ledger to disk."""
        self.ledger.save()
        log.info(f"Alert engine shutdown — total alerts emitted: {self._alert_count}")

    def _write_alerts(self, alerts: List[AlertPayload]):
        date_str = datetime.now().strftime("%Y-%m-%d")
        path = self.alert_dir / f"{date_str}_alerts.jsonl"
        with open(path, "a") as f:
            for a in alerts:
                line = json.dumps(asdict(a), default=str)
                f.write(line + "\n")
                self._alert_count += 1

                # Console alert (colour-coded)
                log.warning(
                    f"🚨 ALERT  |  {a.detected_disease}: {a.confidence_percentage}%  "
                    f"|  Rail pos: {a.rail_position_id}  |  Frame: {a.frame_id}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE-BATCH MODE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
def run_image_mode(args, engine: InferenceEngine, viz: Visualizer, alert_engine: AlertEngine):
    """Process a folder of images or a single image file (offline mode)."""
    source = Path(args.source)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    if source.is_file():
        image_paths = [source]
    elif source.is_dir():
        image_paths = sorted(
            p for p in source.rglob("*") if p.suffix.lower() in exts
        )
    else:
        log.error(f"Source not found: {source}")
        return

    if not image_paths:
        log.error(f"No images found in: {source}")
        return

    log.info(f"Image mode — processing {len(image_paths)} image(s)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_detections = 0
    total_alerts = 0

    for idx, img_path in enumerate(image_paths, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            log.warning(f"[SKIP] Cannot read: {img_path}")
            continue

        # Inference
        detections, _ = engine.infer(frame)
        total_detections += len(detections)

        # Alerts
        alerts = alert_engine.evaluate(detections, rail_position=idx, frame_id=idx)
        total_alerts += len(alerts)

        # Annotate
        annotated = viz.annotate(frame, detections, rail_position=idx)

        # Print summary
        labels = engine.decode_labels(detections)
        label_str = ", ".join(labels) if labels else "No detections (above threshold)"
        rel = img_path.name
        log.info(f"[{idx}/{len(image_paths)}] {rel}  →  {len(detections)} det  |  {label_str}")

        # Save
        out_name = img_path.stem + "_pipeline" + img_path.suffix
        cv2.imwrite(str(OUTPUT_DIR / out_name), annotated)

        # Show if requested
        if args.show:
            cv2.imshow("Greenhouse Pipeline — Image Mode", annotated)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break

    log.info("=" * 60)
    log.info(f"Images processed     : {len(image_paths)}")
    log.info(f"Total detections     : {total_detections}")
    log.info(f"Total alerts fired   : {total_alerts}")
    log.info(f"Results saved to     : {OUTPUT_DIR}")
    log.info(f"Alert logs at        : {ALERT_DIR}")

    if args.show:
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE / VIDEO STREAM MODE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
def run_live_mode(args, engine: InferenceEngine, viz: Visualizer, alert_engine: AlertEngine):
    """Real-time processing from camera or RTSP stream."""
    stream = VideoStreamCapture(source=args.source)
    if not stream.start():
        log.error("Could not start video stream. Exiting.")
        return

    res = stream.resolution
    log.info(f"Stream active at {res[0]}x{res[1]}")
    log.info("Starting live inference loop...")
    log.info("Controls:  [Q]uit  [S]creenshot  [+/-] Conf  [T]oggle HUD  [P]ause")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conf_threshold = engine.conf_threshold
    screenshot_count = 0
    rail_position = 0
    paused = False
    prev_time = time.time()
    fps = 0.0

    # Inference throttle for alerts (avoid flooding)
    last_alert_frame = -10

    try:
        while True:
            if not paused:
                ret, frame = stream.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # Update rail position (auto-increment per batch; replace with serial input)
                if stream.frame_count % 30 == 0:
                    rail_position += 1

                # Inference
                engine.conf_threshold = conf_threshold
                detections, _ = engine.infer(frame)

                # Alerts (throttle: only evaluate every 15 frames to avoid flooding)
                if stream.frame_count - last_alert_frame >= 15 and detections:
                    alert_engine.evaluate(detections, rail_position, stream.frame_count)
                    last_alert_frame = stream.frame_count

                # FPS
                now = time.time()
                fps = 0.8 * fps + 0.2 / max(now - prev_time, 0.001)
                prev_time = now

                # Annotate
                annotated = viz.annotate(
                    frame, detections,
                    fps=fps,
                    conf_threshold=conf_threshold,
                    rail_position=rail_position,
                    paused=paused,
                )
            else:
                # While paused, keep showing the last annotated frame
                pass

            cv2.imshow("Greenhouse Pipeline — Live", annotated)

            # ── Key handling ────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                log.info("Quit requested.")
                break
            elif key == ord("s"):
                screenshot_count += 1
                fname = f"pipeline_screenshot_{screenshot_count:04d}.jpg"
                fpath = OUTPUT_DIR / fname
                cv2.imwrite(str(fpath), annotated)
                log.info(f"Screenshot saved: {fpath}")
            elif key in (ord("+"), ord("=")):
                conf_threshold = min(conf_threshold + 0.05, 0.99)
                log.info(f"Confidence threshold → {conf_threshold*100:.0f}%")
            elif key in (ord("-"), ord("_")):
                conf_threshold = max(conf_threshold - 0.05, 0.05)
                log.info(f"Confidence threshold → {conf_threshold*100:.0f}%")
            elif key == ord("t"):
                viz.show_hud = not viz.show_hud
                log.info(f"HUD {'ON' if viz.show_hud else 'OFF'}")
            elif key == ord("p"):
                paused = not paused
                log.info(f"{'PAUSED' if paused else 'RESUMED'}")

    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl+C).")
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        log.info(f"Frames processed: {stream.frame_count}")
        log.info(f"Total alerts: {alert_engine.alert_count}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI & ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="greenhouse_pipeline",
        description="Greenhouse Precision Agriculture — Rail Camera CV Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python greenhouse_pipeline.py --source 0                  # laptop webcam
  python greenhouse_pipeline.py --source rtsp://ip/stream   # rail camera
  python greenhouse_pipeline.py --source ./test --mode images
  python greenhouse_pipeline.py --source leaf.jpg --mode images --show
        """,
    )
    p.add_argument("--source", type=str, default=DEFAULT_SOURCE,
                   help="Camera index (0), RTSP URL, image file, or image directory. Default: 0")
    p.add_argument("--mode", type=str, choices=["live", "images"], default="live",
                   help="'live' for camera/RTSP stream, 'images' for batch processing. Default: live")
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                   help=f"Confidence threshold (0-1). Default: {CONF_THRESHOLD}")
    p.add_argument("--model", type=str, default=str(MODEL_PATH),
                   help=f"Path to YOLO .pt weights. Default: {MODEL_PATH}")
    p.add_argument("--imgsz", type=int, default=IMG_SIZE,
                   help=f"Inference image size. Default: {IMG_SIZE}")
    p.add_argument("--show", action="store_true",
                   help="(Images mode) Display each annotated image in a window")
    p.add_argument("--no-hud", action="store_true",
                   help="Start with HUD overlay disabled")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Override globals from args
    global IMG_SIZE
    IMG_SIZE = args.imgsz

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   GREENHOUSE PRECISION AGRICULTURE — CV PIPELINE           ║")
    print("║   Rockmelon Leaf Disease Detection System                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Initialise modules
    engine       = InferenceEngine(model_path=args.model, conf_threshold=args.conf)
    viz          = Visualizer(class_names=engine.class_names, show_hud=not args.no_hud)
    alert_engine = AlertEngine()

    log.info(f"Mode: {args.mode.upper()}")
    log.info(f"Source: {args.source}")

    try:
        if args.mode == "images":
            run_image_mode(args, engine, viz, alert_engine)
        else:
            run_live_mode(args, engine, viz, alert_engine)
    finally:
        alert_engine.shutdown()

    print()
    log.info("Pipeline shutdown complete.")


if __name__ == "__main__":
    main()
