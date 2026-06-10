import os
from collections import Counter

glass_dir = r"d:\Code\Code\DataSetProject\DataSet\glass"
files = [f for f in os.listdir(glass_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Print some file names containing keywords
keywords = ['broken', 'shard', 'crushed', 'piece', 'green', 'brown', 'white', 'bottle', 'cup']
keyword_counts = Counter()

for f in files:
    for kw in keywords:
        if kw in f.lower():
            keyword_counts[kw] += 1

print(f"Total glass images: {len(files)}")
print("\nKeyword counts in glass filenames:")
for kw, cnt in keyword_counts.items():
    print(f"  '{kw}': {cnt} files")

print("\nSample glass filenames containing 'green':")
green_files = [f for f in files if 'green' in f.lower()]
print(green_files[:10])

print("\nSample glass filenames containing 'brown':")
brown_files = [f for f in files if 'brown' in f.lower()]
print(brown_files[:10])
