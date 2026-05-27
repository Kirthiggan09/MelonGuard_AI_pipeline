import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2

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

# Global state
stream = None
engine = None
viz = None
alert_engine = None

# A thread-safe list to hold recent alerts for the dashboard
recent_alerts = []
alerts_lock = threading.Lock()

# Custom AlertEngine wrapper to capture alerts in memory
class MemoryAlertEngine(AlertEngine):
    def evaluate(self, detections, rail_position, frame_id):
        alerts = super().evaluate(detections, rail_position, frame_id)
        if alerts:
            with alerts_lock:
                for a in alerts:
                    # add to recent alerts list
                    from dataclasses import asdict
                    recent_alerts.insert(0, asdict(a))
                # keep only the last 100 alerts in memory
                while len(recent_alerts) > 100:
                    recent_alerts.pop()
        return alerts

def generate_frames():
    """Generator function that continuously yields MJPEG frames."""
    global stream, engine, viz, alert_engine
    
    rail_position = 0
    fps = 0.0
    prev_time = time.time()
    last_alert_frame = -15
    conf_threshold = engine.conf_threshold

    while True:
        ret, frame = stream.read()
        if not ret:
            time.sleep(0.05)
            continue

        # Auto-increment rail position (simulated)
        if stream.frame_count % 30 == 0:
            rail_position += 1

        # Inference
        detections, _ = engine.infer(frame)

        # Alerts (throttle: only evaluate every 15 frames)
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
