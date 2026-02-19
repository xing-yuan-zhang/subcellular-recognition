import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from datasets import image_transform


def cache_images_to_tensor(input_json, image_dir, output_dir, output_index_path=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json, 'r') as f:
        data_list = json.load(f)

    for idx, item in tqdm(enumerate(data_list), total=len(data_list), desc="Preprocessing images"):
        img_path = os.path.join(image_dir, item['filename'])
        img = Image.open(img_path).convert('RGB')
        tensor = image_transform(img)
        torch.save(tensor, os.path.join(output_dir, f"{idx}.pt"))

    labels = [int(item['label']) for item in data_list]
    index_path = output_index_path if output_index_path else os.path.join(output_dir, 'index_label.json')
    with open(index_path, 'w') as f:
        json.dump(labels, f)
