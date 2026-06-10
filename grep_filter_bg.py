import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("app.py", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if "filter_background:" in line or "filter_background" in line:
            print(f"{idx}: {line.strip()}")
