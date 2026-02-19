import os
import json
import pandas as pd


def generate_label_tasks(image_dir, output_json):
    tasks = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith('.png'):
            tasks.append({
                "data": {
                    "image": f"/data/local-files/?d=images/{fname}"
                }
            })
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(tasks, f, indent=2)
    print(f"-----Generated {len(tasks)} labeling tasks, saved to {output_json}-----")


def combine_labels(csv_path, output_json, image_dir):
    df = pd.read_csv(csv_path)
    df['filename'] = df['image'].apply(lambda path: os.path.basename(str(path)))
    data_list = []
    for _, row in df.iterrows():
        data_list.append({
            "filename": row['filename'],
            "label": int(row['label'])
        })
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(data_list, f, indent=2)
    print(f"-----Saved labeled data JSON to {output_json}, {len(data_list)} entries-----")
    all_images = {f for f in os.listdir(image_dir) if f.lower().endswith('.png')}
    labeled_images = set(df['filename'].tolist())
    unlabeled = all_images - labeled_images
    for fname in unlabeled:
        os.remove(os.path.join(image_dir, fname))
    if unlabeled:
        print(f"Removed {len(unlabeled)} unlabeled images from {image_dir}")
