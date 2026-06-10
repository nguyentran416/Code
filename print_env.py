import os

print("--- Environment Variables ---")
for k, v in os.environ.items():
    if any(keyword in k for keyword in ['DETECT', 'SALIENCY', 'SINGLE']):
        print(f"  {k} = {v}")
