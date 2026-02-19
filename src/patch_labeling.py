import os
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def label_patches_interactive(patch_dir):
    label_file = os.path.join(patch_dir, 'labels.csv')
    if os.path.exists(label_file):
        df = pd.read_csv(label_file)
        labeled_files = set(df['filename'].tolist())
    else:
        df = pd.DataFrame(columns=['filename', 'label'])
        labeled_files = set()
    all_files = sorted([f for f in os.listdir(patch_dir) if f.lower().endswith(('.tif', '.tiff'))])
    stop = False
    for fname in all_files:
        if fname in labeled_files:
            continue
        img = tifffile.imread(os.path.join(patch_dir, fname))
        plt.imshow(img, cmap='gray')
        plt.title(fname)
        plt.axis('off')
        plt.show()

        while True:
            label = input("-----Label this image (0 = False, 1 = True, q = quit)-----")
            if label == 'q':
                stop = True
                break
            if label in ['0', '1']:
                df.loc[len(df)] = [fname, int(label)]
                break
            else:
                print("-----Invalid input-----")
        if stop:
            break

    df.to_csv(label_file, index=False)
