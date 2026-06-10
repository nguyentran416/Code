import sys
import os

sys.path.append(os.getcwd())
import app

bbox = [0.0, 0.06315789473684211, 0.9388888888888889, 1.0]
margin = app.DETECT_EDGE_MARGIN

print(f"DETECT_EDGE_MARGIN in app: {margin}")

xmin, ymin, xmax, ymax = [float(v) for v in bbox]
spans_horizontal = (xmin <= margin) and (xmax >= 1.0 - margin)
spans_vertical   = (ymin <= margin) and (ymax >= 1.0 - margin)

print(f"xmin={xmin} <= margin={margin}: {xmin <= margin}")
print(f"xmax={xmax} >= 1.0 - margin={1.0 - margin}: {xmax >= 1.0 - margin}")
print(f"spans_horizontal: {spans_horizontal}")

print(f"ymin={ymin} <= margin={margin}: {ymin <= margin}")
print(f"ymax={ymax} >= 1.0 - margin={1.0 - margin}: {ymax >= 1.0 - margin}")
print(f"spans_vertical: {spans_vertical}")

print(f"is_edge_box(bbox): {app.is_edge_box(bbox)}")
