with open("test_conditional_saliency.py", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if "filter_background" in line:
            print(f"{idx}: {line.strip()}")
