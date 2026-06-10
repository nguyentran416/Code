import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

search_dir = r"d:\Code"
print("Searching for 'or corner box' in all files under d:\\Code:")
for root, dirs, files in os.walk(search_dir):
    if any(p in root for p in ["venv", ".venv", ".git", "tfenv"]):
        continue
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    for idx, line in enumerate(file, 1):
                        if "or corner box" in line:
                            print(f"{path}:{idx} -> {line.strip()}")
            except:
                pass
