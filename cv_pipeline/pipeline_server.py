import os
import sys
import json
import time
import argparse
import threading
import queue
import re
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, send_from_directory, request
from flask_cors import CORS
import cv2
import serial

# ---------- Hardware Control Config & State ----------
DEFAULT_SERIAL = "/dev/ttyTHS1"
DEFAULT_BAUD = 115200
DEADZONE_FRAC = 0.25

serial_port = None
serial_status = "disconnected"
auto_control = False
last_command = None
real_rail_position = 0

send_queue = queue.Queue()
rail_logs = []

def log_rail(msg):
    ts = time.strftime('%H:%M:%S')
    rail_logs.append(f"{ts} {msg}")
    if len(rail_logs) > 200:
        rail_logs.pop(0)

def connect_serial(port, baud):
    global serial_port, serial_status
    try:
        serial_port = serial.Serial(port, baud, timeout=0.1)
        serial_status = f"connected ({port}@{baud})"
        log_rail(f"Serial connected {port}@{baud}")
        threading.Thread(target=serial_reader_thread, daemon=True).start()
        return True
    except Exception as e:
        log_rail(f"Serial open error: {e}")
        return False

def disconnect_serial():
    global serial_port, serial_status
    try:
        if serial_port and serial_port.is_open:
            serial_port.close()
    except:
        pass
    serial_port = None
    serial_status = "disconnected"
    log_rail("Serial disconnected")

def serial_reader_thread():
    global serial_port, real_rail_position
    while True:
        try:
            if serial_port and serial_port.is_open:
                if serial_port.in_waiting:
                    line = serial_port.readline().decode(errors="ignore").strip()
                    if line:
                        log_rail(f"<-- {line}")
                        match = re.search(r'Pos:\s*(\d+)', line)
                        if match:
                            real_rail_position = int(match.group(1))
                else:
                    time.sleep(0.05)
            else:
                time.sleep(0.5)
        except Exception as e:
            log_rail(f"[ERR] Serial read error: {e}")
            time.sleep(0.5)

def send_worker():
    global last_command
    while True:
        try:
            cmd = send_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            if serial_port and serial_port.is_open:
                serial_port.write((cmd + "\n").encode())
                log_rail(f"--> {cmd}")
            else:
                log_rail(f"(not connected) Would send: {cmd}")
            last_command = cmd
        except Exception as e:
            log_rail(f"[ERR send] {e}")

# Import our pipeline modules
from greenhouse_pipeline import (
    VideoStreamCapture,
    InferenceEngine,
    Visualizer,
    AlertEngine,
    CONF_THRESHOLD,
    IMG_SIZE,
    DEFAULT_SOURCE,
    ALERT_DIR,
    log
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes so the Node.js dashboard can access it

# ── Disease Capture Config ──────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = SCRIPT_DIR / "disease_captures"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
CAPTURE_COOLDOWN_SEC = 5.0    # Minimum seconds between captures
MAX_CAPTURES = 200            # Keep the folder from growing unbounded

# Global state
stream = None
engine = None
viz = None
alert_engine = None
last_capture_time = 0.0       # Timestamp of the last saved capture

# A thread-safe list to hold recent alerts for the dashboard
recent_alerts = []
alerts_lock = threading.Lock()

# Line Scan tracking state
scan_active = False
scan_summary = {}
scan_lock = threading.Lock()

# Custom AlertEngine wrapper to capture alerts in memory
class MemoryAlertEngine(AlertEngine):
    def evaluate(self, detections, rail_position, frame_id):
        alerts = super().evaluate(detections, rail_position, frame_id)
        if alerts:
            global scan_active, scan_summary
            # Add to scan summary if active
            with scan_lock:
                if scan_active:
                    for a in alerts:
                        disease = a.detected_disease
                        if disease not in scan_summary:
                            scan_summary[disease] = {"count": 0, "total_conf": 0.0}
                        scan_summary[disease]["count"] += 1
                        scan_summary[disease]["total_conf"] += a.confidence_percentage

            with alerts_lock:
                for a in alerts:
                    # add to recent alerts list
                    from dataclasses import asdict
                    recent_alerts.insert(0, asdict(a))
                # keep only the last 100 alerts in memory
                while len(recent_alerts) > 100:
                    recent_alerts.pop()
        return alerts

def _save_disease_capture(annotated_frame, detections):
    """Save an annotated frame to the disease_captures folder."""
    global last_capture_time
    now = time.time()
    if now - last_capture_time < CAPTURE_COOLDOWN_SEC:
        return  # Throttle: skip if we captured recently

    # Build a descriptive filename from the top detection
    top = max(detections, key=lambda d: d.confidence)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    conf_pct = int(top.confidence * 100)
    fname = f"{ts}_{top.display_name}_{conf_pct}pct.jpg"

    cv2.imwrite(str(CAPTURE_DIR / fname), annotated_frame)
    last_capture_time = now
    log.info(f"📸 Disease capture saved: {fname}")

    # Also save to the external categorized OneDrive folder
    onedrive_base = Path(r"C:\Users\Kirthiggan\OneDrive\Documents\rockmelon lead disease (pictures)")
    try:
        # Create a specific folder for this disease category
        category_dir = onedrive_base / top.display_name
        category_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(category_dir / fname), annotated_frame)
    except Exception as e:
        log.error(f"Failed to save image to OneDrive categorized folder: {e}")

    # Prune old captures if folder exceeds MAX_CAPTURES
    captures = sorted(CAPTURE_DIR.glob("*.jpg"), key=os.path.getmtime)
    while len(captures) > MAX_CAPTURES:
        oldest = captures.pop(0)
        oldest.unlink(missing_ok=True)


def generate_frames():
    """Generator function that continuously yields MJPEG frames."""
    global stream, engine, viz, alert_engine, real_rail_position, auto_control
    
    fps = 0.0
    prev_time = time.time()
    last_alert_frame = -15
    conf_threshold = engine.conf_threshold

    while True:
        ret, frame = stream.read()
        if not ret:
            time.sleep(0.05)
            continue

        # Inference
        detections, _ = engine.infer(frame)

        # Auto control based on detections
        if auto_control and detections:
            best = max(detections, key=lambda d: d.confidence)
            frame_w = frame.shape[1]
            cx = int((best.box[0] + best.box[2]) / 2)
            left_thresh = frame_w * (0.5 - DEADZONE_FRAC / 2)
            right_thresh = frame_w * (0.5 + DEADZONE_FRAC / 2)
            
            # Send centering command every 5 frames to avoid flooding
            if stream.frame_count % 5 == 0:
                if cx < left_thresh:
                    send_queue.put('l')
                elif cx > right_thresh:
                    send_queue.put('r')

        # Alerts (throttle: only evaluate every 15 frames)
        if stream.frame_count - last_alert_frame >= 15 and detections:
            alert_engine.evaluate(detections, real_rail_position, stream.frame_count)
            last_alert_frame = stream.frame_count

        # ── Auto-capture disease screenshots ────────────────────────────
        risk_detections = [d for d in detections if d.is_risk]
        if risk_detections:
            _save_disease_capture(frame, risk_detections)

        # FPS
        now = time.time()
        fps = 0.8 * fps + 0.2 / max(now - prev_time, 0.001)
        prev_time = now

        # Annotate
        annotated = viz.annotate(
            frame, detections,
            fps=fps,
            conf_threshold=conf_threshold,
            rail_position=real_rail_position,
            paused=False,
        )

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/alerts')
def api_alerts():
    """Returns the most recent alerts as JSON."""
    with alerts_lock:
        return jsonify({
            "status": "success",
            "count": len(recent_alerts),
            "alerts": recent_alerts
        })

@app.route('/api/status')
def api_status():
    """Returns system status."""
    return jsonify({
        "status": "running",
        "fps": 0, # could add real fps here
        "source": stream.source,
        "frame_count": stream.frame_count
    })

@app.route('/api/captures')
def api_captures():
    """Returns a list of saved disease capture filenames (newest first)."""
    captures = sorted(CAPTURE_DIR.glob("*.jpg"), key=os.path.getmtime, reverse=True)
    items = []
    for p in captures[:50]:  # Return up to 50 most recent
        parts = p.stem.split("_")
        # Filename format: 20260721_231800_Fungus_89pct.jpg
        disease = parts[2] if len(parts) >= 4 else "Unknown"
        conf = parts[3].replace("pct", "%") if len(parts) >= 4 else ""
        items.append({
            "filename": p.name,
            "disease": disease,
            "confidence": conf,
            "timestamp": datetime.fromtimestamp(p.stat().st_mtime).isoformat()
        })
    return jsonify({"status": "success", "count": len(items), "captures": items})

@app.route('/api/scan/start', methods=['POST'])
def api_scan_start():
    """Starts tracking a new line scan."""
    global scan_active, scan_summary
    with scan_lock:
        scan_active = True
        scan_summary = {}
    return jsonify({"status": "success", "message": "Scan started"})

@app.route('/api/scan/stop', methods=['POST'])
def api_scan_stop():
    """Stops tracking and returns the aggregated summary of diseases."""
    global scan_active, scan_summary
    with scan_lock:
        scan_active = False
        results = []
        for disease, data in scan_summary.items():
            count = data["count"]
            avg_conf = data["total_conf"] / count if count > 0 else 0
            results.append({
                "disease": disease,
                "count": count,
                "avg_confidence": round(avg_conf, 1)
            })
    return jsonify({"status": "success", "summary": results})

@app.route('/captures/<path:filename>')
def serve_capture(filename):
    """Serve a saved disease capture image."""
    return send_from_directory(str(CAPTURE_DIR), filename)

@app.route('/api/hardware/status')
def api_hardware_status():
    return jsonify({
        'serial_status': serial_status,
        'auto_control': auto_control,
        'last_command': last_command,
        'real_rail_position': real_rail_position,
        'rail_logs': rail_logs[-20:]
    })

@app.route('/api/hardware/connect', methods=['POST'])
def api_hardware_connect():
    data = request.json
    port = data.get('port', DEFAULT_SERIAL)
    baud = int(data.get('baud', DEFAULT_BAUD))
    success = connect_serial(port, baud)
    return jsonify({'success': success})

@app.route('/api/hardware/disconnect', methods=['POST'])
def api_hardware_disconnect():
    disconnect_serial()
    return jsonify({'success': True})

@app.route('/api/hardware/send', methods=['POST'])
def api_hardware_send():
    data = request.json
    cmd = data.get('cmd')
    if cmd:
        send_queue.put(cmd)
    return jsonify({'success': True})

@app.route('/api/hardware/toggle_auto', methods=['POST'])
def api_hardware_toggle_auto():
    global auto_control
    auto_control = not auto_control
    log_rail(f"Auto control {'ENABLED' if auto_control else 'DISABLED'}")
    return jsonify({'success': True, 'auto_control': auto_control})

def start_pipeline(source=DEFAULT_SOURCE, conf=CONF_THRESHOLD):
    global stream, engine, viz, alert_engine
    
    log.info(f"Starting pipeline server with source: {source}")
    
    engine = InferenceEngine(conf_threshold=conf)
    viz = Visualizer(class_names=engine.class_names, show_hud=True)
    alert_engine = MemoryAlertEngine()
    
    stream = VideoStreamCapture(source=source)
    if not stream.start():
        log.error("Failed to start video stream.")
        sys.exit(1)
        
    threading.Thread(target=send_worker, daemon=True).start()
    log.info("Pipeline started successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Greenhouse Pipeline Flask Server")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Camera index or RTSP URL")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--port", type=int, default=5000, help="Port to run Flask server on")
    args = parser.parse_args()

    start_pipeline(source=args.source, conf=args.conf)
    
    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    finally:
        if alert_engine:
            alert_engine.shutdown()
        if stream:
            stream.stop()
