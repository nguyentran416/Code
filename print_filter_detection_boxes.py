import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for idx in range(745, 795):
        if idx < len(lines):
            print(f"{idx+1}: {lines[idx].rstrip()}")
