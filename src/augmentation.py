import os
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
import matplotlib.pyplot as plt


def augment_image(img_array):
    augmented = [img_array]
    augmented.append(np.fliplr(img_array))
    pil_img = Image.fromarray(img_array)
    bright_img = ImageEnhance.Brightness(pil_img).enhance(random.uniform(0.7, 1.3))
    augmented.append(np.array(bright_img))
    contrast_img = ImageEnhance.Contrast(pil_img).enhance(random.uniform(0.8, 1.4))
    augmented.append(np.array(contrast_img))

    noise = img_array + np.random.normal(0, 10, img_array.shape)
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    augmented.append(noise)

    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D((img_array.shape[1]//2, img_array.shape[0]//2), angle, 1.0)
        rotated = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))
        augmented.append(rotated)
    return augmented


def augment_dataset(input_json, image_dir, output_dir, output_json=None):
    os.makedirs(output_dir, exist_ok=True)
    if output_json is None:
        output_json = os.path.join(output_dir, 'data_augmented.json')

    with open(input_json, 'r') as f:
        data = json.load(f)
    augmented_data = []
    for item in data:
        fname = item['filename']
        label = int(item['label'])
        fpath = os.path.join(image_dir, fname)
        if not os.path.exists(fpath):
            continue

        img = np.array(Image.open(fpath).convert('L'))
        aug_images = augment_image(img)
        base_name = os.path.splitext(fname)[0]

        for i, aug_img in enumerate(aug_images):
            rgb_img = np.stack([aug_img]*3, axis=-1)
            new_name = f"{base_name}_aug{i}.png"
            Image.fromarray(rgb_img).save(os.path.join(output_dir, new_name))
            augmented_data.append({"filename": new_name, "label": label})

    with open(output_json, 'w') as f:
        json.dump(augmented_data, f, indent=2)


def show_augmentations(img_array):
    aug_imgs = augment_image(img_array)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Augmentations", fontsize=16)
    for i, ax in enumerate(axes.flatten()):
        if i < len(aug_imgs):
            ax.imshow(aug_imgs[i], cmap='gray')
        ax.axis('off')
    plt.show()
