"""
process_wlasl.py — Process WLASL `.mp4` videos into Mediapipe sequences (X.npy, y.npy)
for LSTM training.

Usage:
    python process_wlasl.py --subset 10   # Processes top 10 labels
    python process_wlasl.py --all         # Processes all labels (takes a LONG time)
"""

import sys
import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm

# The absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))

from app.preprocessing.hand_detection import extract_landmarks, SEQUENCE_LENGTH
WLASL_JSON_PATH = os.path.join(BASE_DIR, "dataset/archive/wlasl-complete/WLASL_v0.3.json")
VIDEOS_DIR = os.path.join(BASE_DIR, "dataset/archive/wlasl-complete/videos")
MODELS_DIR = os.path.join(BASE_DIR, "../app/models")

def extract_sliding_windows(sequence, window_size=SEQUENCE_LENGTH, step=5):
    """Generate multiple overlapping windows from a sequence."""
    seq_len = len(sequence)
    if seq_len < window_size:
        # If too short, pad it to window_size and return just one
        pad_len = window_size - seq_len
        padding = [sequence[-1]] * pad_len
        return [sequence + padding]
        
    windows = []
    # Slide window across sequence
    for i in range(0, seq_len - window_size + 1, step):
        windows.append(sequence[i : i + window_size])
        
    return windows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=10, help="Number of labels to process (default: 10)")
    parser.add_argument("--all", action="store_true", help="Process all labels (Overwrites subset)")
    args = parser.parse_args()

    # Load JSON
    with open(WLASL_JSON_PATH, "r") as f:
        wlasl_data = json.load(f)

    # Target: ONLY alphabets (length 1)
    wlasl_data = [s for s in wlasl_data if len(s["gloss"].strip()) == 1]
    print(f"[Process] Selected {len(wlasl_data)} alphabets.")

    X_all = []
    y_all = []
    label_map = {}
    
    print(f"Processing {len(wlasl_data)} classes...")

    for label_id, class_info in enumerate(tqdm(wlasl_data, desc="Classes")):
        gloss = class_info["gloss"]
        label_map[label_id] = gloss
        
        for instance in class_info["instances"]:
            video_id = instance["video_id"]
            video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
            
            if not os.path.exists(video_path):
                continue
                
            cap = cv2.VideoCapture(video_path)
            sequence = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Extract landmarks
                lm = extract_landmarks(frame)
                if lm is not None:
                    sequence.append(lm)
                    
            cap.release()
            
            # Use industrial sliding window augmentation
            if len(sequence) > 0:
                sampled_windows = extract_sliding_windows(sequence, SEQUENCE_LENGTH)
                for win in sampled_windows:
                    X_all.append(np.array(win))
                    y_all.append(label_id)

    if len(X_all) == 0:
        print("Error: No valid sequences processed. Did you extract the videos?")
        return

    X_out = np.array(X_all, dtype=np.float32)
    y_out = np.array(y_all, dtype=np.int32)
    
    np.save(os.path.join(BASE_DIR, "X.npy"), X_out)
    np.save(os.path.join(BASE_DIR, "y.npy"), y_out)
    
    # Save labels.json
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "labels.json"), "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f)
        
    print(f"\nProcessing complete!")
    print(f"Saved X.npy: {X_out.shape}")
    print(f"Saved y.npy: {y_out.shape}")
    print(f"Saved labels.json to app/models/labels.json")

if __name__ == "__main__":
    main()
