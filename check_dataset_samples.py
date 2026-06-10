import os

dataset_dir = r"d:\Code\Code\DataSetProject\DataSet"
classes = os.listdir(dataset_dir)

print("Dataset classes and image counts:")
for cls in classes:
    cls_path = os.path.join(dataset_dir, cls)
    if os.path.isdir(cls_path):
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        print(f"  {cls}: {len(files)} images")
        if len(files) > 0:
            print(f"    Sample filenames: {files[:5]}")
