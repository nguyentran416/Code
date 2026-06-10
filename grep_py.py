with open("app.py", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if "DETECT_EDGE_MARGIN" in line:
            print(f"{idx}: {line.strip()}")
