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
detection_model = app.load_detection_model()

# Image paths
images = {
    "1. Guava + Bottle": r"C:\Users\hiii\.gemini\antigravity-ide\brain\b00bea7c-d167-448c-b493-a5c60f0df296\media__1781089108996.jpg",
    "2. Notebook + Bottle": r"C:\Users\hiii\.gemini\antigravity-ide\brain\b00bea7c-d167-448c-b493-a5c60f0df296\media__1781089327148.png",
    "3. Underpants + Bottle": r"C:\Users\hiii\.gemini\antigravity-ide\brain\b00bea7c-d167-448c-b493-a5c60f0df296\media__1781090086571.jpg",
    "4. Red Shirt + Paper": r"C:\Users\hiii\.gemini\antigravity-ide\brain\b00bea7c-d167-448c-b493-a5c60f0df296\media__1781090114553.jpg"
}

EXCLUDED_COCO_CLASSES = {
    1,   # person
    2,   # bicycle
    3,   # car
    4,   # motorcycle
    5,   # airplane
    6,   # bus
    7,   # train
    8,   # truck
    9,   # boat
    15,  # bench
    27,  # backpack
    31,  # handbag
    33,  # suitcase
    62,  # chair
    63,  # couch
    64,  # potted plant
    65,  # bed
    67,  # dining table
    70,  # toilet
    72,  # tv
    78,  # microwave
    79,  # oven
    80,  # toaster
    81,  # sink
    82,  # refrigerator
}

def detect_objects_ssd_tuned(image_path):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    arr = np.array(img)
    inp = tf.convert_to_tensor(arr, dtype=tf.uint8)[tf.newaxis, ...]
    sig = detection_model.signatures['serving_default']
    out = sig(inp)
    
    boxes = out['detection_boxes'][0].numpy()
    scores = out['detection_scores'][0].numpy()
    classes = out['detection_classes'][0].numpy().astype(int)
    num = int(out['num_detections'][0].numpy())
    
    min_pixel = 56
    detections = []
    for i in range(min(100, num)):
        score = float(scores[i])
        if score < app.DETECT_MIN_SCORE:
            continue
        
        cls_id = int(classes[i])
        if cls_id in EXCLUDED_COCO_CLASSES:
            continue
            
        ymin, xmin, ymax, xmax = boxes[i].tolist()
        box_area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
        if box_area < app.DETECT_MIN_BOX_AREA or box_area > app.DETECT_MAX_BOX_AREA:
            continue
            
        bw = max(0.0, xmax - xmin)
        bh = max(0.0, ymax - ymin)
        if bh < 1e-6:
            continue
        aspect = bw / bh
        if aspect < app.DETECT_MIN_ASPECT or aspect > app.DETECT_MAX_ASPECT:
            continue
            
        norm_bbox_check = [xmin, ymin, xmax, ymax]
        if app.is_edge_box(norm_bbox_check):
            continue
            
        left, top, right, bottom = app._expand_ssd_box(ymin, xmin, ymax, xmax, width, height)
        if (right - left) < min_pixel or (bottom - top) < min_pixel:
            continue
            
        crop = app.upscale_crop_if_needed(img.crop((left, top, right, bottom)))
        label, label_score = app.classify_crop_robust(crop)
        combined = score * label_score
        if not label or label_score < app.DETECT_MIN_LABEL_SCORE or combined < app.DETECT_MIN_COMBINED_SCORE:
            continue
            
        norm_bbox = [left/width, top/height, right/width, bottom/height]
        cb = app.center_bias_score(norm_bbox)
        effective_score = label_score * cb
        
        detections.append({
            'bbox': norm_bbox,
            'score': effective_score,
            'label': label,
            'label_score': label_score,
            'center_bias': cb,
            'source': 'ssd',
        })
    if detections:
        detections = app.nms_detections(detections)
    detections.sort(key=lambda d: d.get('score', 0.0), reverse=True)
    return detections[:app.MAX_MULTI_DETECTIONS]

def run_tuned_pipeline(name, path):
    print(f"\n==================================================")
    print(f"Results for: {name}")
    print(f"==================================================")
    
    # Run SSD
    ssd_dets = detect_objects_ssd_tuned(path)
    print(f"SSD Detections ({len(ssd_dets)}):")
    for d in ssd_dets:
        print(f"  - {d['label']} ({d['label_score']:.2f}) at {[round(c, 2) for c in d['bbox']]}")
        
    # Saliency run condition: SSD < 2
    detections = list(ssd_dets)
    if len(detections) < 2:
        print(f"SSD detections count ({len(detections)}) < 2, running Saliency...")
        # Run Saliency with 0.60 threshold
        original_threshold = app.SALIENCY_MIN_LABEL_SCORE
        app.SALIENCY_MIN_LABEL_SCORE = 0.60
        try:
            saliency_dets = app.detect_objects_saliency(path)
        finally:
            app.SALIENCY_MIN_LABEL_SCORE = original_threshold
            
        print(f"Saliency detected ({len(saliency_dets)}):")
        for d in saliency_dets:
            print(f"  - {d['label']} ({d['label_score']:.2f}) at {[round(c, 2) for c in d['bbox']]}")
        detections = app.nms_detections(detections + saliency_dets)
    else:
        print(f"SSD detections count ({len(detections)}) >= 2, skipping Saliency.")
        
    detections = app.filter_detection_boxes(detections)
    detections = app.filter_background_detections(detections)
    detections = app.nms_detections(detections)[:app.MAX_MULTI_DETECTIONS]
    
    print(f"\nFinal Consolidated Detections ({len(detections)}):")
    for d in detections:
        print(f"  * {d['label']} ({d['label_score']:.2f}) from {d.get('source')} at {[round(c, 2) for c in d['bbox']]}")

for name, path in images.items():
    run_tuned_pipeline(name, path)
