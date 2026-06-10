import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

project_dir = r"d:\Code\Code\Project\Waste-Classification-Web-main"

print("Searching for 'loại box' or 'corner box' in all .py files:")
for root, dirs, files in os.walk(project_dir):
    if "venv" in root or ".venv" in root:
        continue
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    for idx, line in enumerate(file, 1):
                        if any(w in line for w in ["loại box", "corner box"]):
                            print(f"{os.path.relpath(path, project_dir)}:{idx} -> {line.strip()}")
            except Exception as e:
                pass
