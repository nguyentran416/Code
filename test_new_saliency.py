import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

# Import functions from app.py
sys.path.append(os.getcwd())
import app

app.ensure_model_ready()
app.load_detection_model()

img1_path = r"C:\Users\hiii\.gemini\antigravity-ide\brain\b00bea7c-d167-448c-b493-a5c60f0df296\media__1781090086571.jpg"
img2_path = r"C:\Users\hiii\.gemini\antigravity-ide\brain\b00bea7c-d167-448c-b493-a5c60f0df296\media__1781090114553.jpg"

def analyze_image(path):
    print(f"\n==================================================")
    print(f"Analyzing: {os.path.basename(path)}")
    print(f"==================================================")
    
    # Run SSD
    ssd_dets = app.detect_objects_ssd(path)
    print(f"SSD detections ({len(ssd_dets)}):")
    for d in ssd_dets:
        print(f"  - Label: {d['label']}, Score: {d['score']:.4f}, Bbox: {[round(c, 2) for c in d['bbox']]}")
        
    # Run Saliency
    saliency_dets = app.detect_objects_saliency(path)
    print(f"Saliency detections ({len(saliency_dets)}):")
    for d in saliency_dets:
        print(f"  - Label: {d['label']}, Score: {d['score']:.4f}, Bbox: {[round(c, 2) for c in d['bbox']]}")

print("=== IMAGE 1 (Underpants + Bottle) ===")
analyze_image(img1_path)

print("\n=== IMAGE 2 (Red Shirt + Paper) ===")
analyze_image(img2_path)
