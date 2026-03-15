import os
import pandas as pd
import re

image_dir = "../zoon_imgs/"  # adjust to your folder name

rows = []
for filename in os.listdir(image_dir):
    if not filename.endswith(".png"):
        continue
    
    # Parse tile_id and mse from filename
    match = re.match(r"tile_(\d+)_mse_([\d.]+)\.png", filename)
    if match:
        tile_id = int(match.group(1))
        mse = float(match.group(2))
        rows.append({
            "filename": filename,
            "tile_id": tile_id,
            "mse": mse
        })

manifest = pd.DataFrame(rows).sort_values("mse", ascending=False)
manifest.to_csv("manifest.csv", index=False)
print(f"Manifest created: {len(manifest)} subjects")