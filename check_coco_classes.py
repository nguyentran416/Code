import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

# COCO label names mapping
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie',
    33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

import app
app.ensure_model_ready()
detection_model = app.load_detection_model()

img1_path = r"C:\Users\hiii\.gemini\antigravity-ide/brain/b00bea7c-d167-448c-b493-a5c60f0df296/media__1781089108996.jpg"
img2_path = r"C:\Users\hiii\.gemini\antigravity-ide/brain/b00bea7c-d167-448c-b493-a5c60f0df296/media__1781089327148.png"

def analyze_ssd_classes(path):
    print(f"\nAnalyzing SSD classes for: {os.path.basename(path)}")
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    inp = tf.convert_to_tensor(arr, dtype=tf.uint8)[tf.newaxis, ...]
    sig = detection_model.signatures['serving_default']
    out = sig(inp)
    
    boxes = out['detection_boxes'][0].numpy()
    scores = out['detection_scores'][0].numpy()
    classes = out['detection_classes'][0].numpy().astype(int)
    num = int(out['num_detections'][0].numpy())
    
    for i in range(min(10, num)):
        score = float(scores[i])
        if score < 0.35:
            continue
        cls_id = int(classes[i])
        cls_name = COCO_CLASSES.get(cls_id, f"Unknown ({cls_id})")
        ymin, xmin, ymax, xmax = boxes[i].tolist()
        print(f"  Det #{i+1}: Class: {cls_name} ({cls_id}), Score: {score:.4f}, Bbox: {[round(c, 2) for c in [xmin, ymin, xmax, ymax]]}")

print("=== IMAGE 1 (Guava + Bottle) ===")
analyze_ssd_classes(img1_path)

print("\n=== IMAGE 2 (Notebook + Bottle) ===")
analyze_ssd_classes(img2_path)
