import os
import sys

sys.path.append(os.getcwd())
import app

print("app.DETECT_EDGE_MARGIN =", app.DETECT_EDGE_MARGIN)
print("os.environ.get('DETECT_EDGE_MARGIN') =", os.environ.get('DETECT_EDGE_MARGIN'))
