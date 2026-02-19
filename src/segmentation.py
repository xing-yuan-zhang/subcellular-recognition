import os
import tifffile
import numpy as np
from PIL import Image


def segment_full_images(input_dir, output_dir, tile_size=256, full_size=2048):
    os.makedirs(output_dir, exist_ok=True)
    tiles_per_dim = full_size // tile_size
    for idx, filename in enumerate(sorted(os.listdir(input_dir))):
        if not (filename.lower().endswith('.tif') or filename.lower().endswith('.tiff')):
            continue
        image = tifffile.imread(os.path.join(input_dir, filename))
        if image.shape[0] != full_size or image.shape[1] != full_size:
            print(f"Skipped (not {full_size}x{full_size}): {filename}")
            continue

        if image.dtype != np.uint8:
            image = (image / image.max() * 255).astype(np.uint8)
        base_id = f"img{idx:03d}"
        for i in range(tiles_per_dim):
            for j in range(tiles_per_dim):
                patch = image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
                patch_img = Image.fromarray(patch)
                patch_filename = f"{base_id}_{i:02d}_{j:02d}.png"
                patch_img.save(os.path.join(output_dir, patch_filename))
