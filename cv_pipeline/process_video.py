import sys
import cv2
import argparse
from pathlib import Path

from greenhouse_pipeline import InferenceEngine, Visualizer, AlertEngine

def process_video(source_path, output_path, conf_threshold):
    engine = InferenceEngine(conf_threshold=conf_threshold)
    viz = Visualizer(class_names=engine.class_names, show_hud=True)
    
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"Failed to open {source_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # Check for NaN
        fps = 30.0
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    
    print(f"Processing {source_path}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # Inference
        detections, _ = engine.infer(frame)
        total_detections += len(detections)
        
        # Annotate
        annotated = viz.annotate(
            frame, detections,
            fps=fps,
            conf_threshold=conf_threshold,
            rail_position=0,
            paused=False,
        )
        
        out.write(annotated)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, Detections so far: {total_detections}", flush=True)
            
    cap.release()
    out.release()
    print(f"Finished! Processed {frame_count} frames. Total detections: {total_detections}")
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    
    process_video(args.source, args.output, args.conf)
