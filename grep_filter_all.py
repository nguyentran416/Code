import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("app.py", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if any(w in line for w in ["or corner box", "filter_background", "edge box"]):
            print(f"{idx}: {line.strip()}")
