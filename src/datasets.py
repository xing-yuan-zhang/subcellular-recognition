import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


class RGBImageDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(os.path.join(self.img_dir, item['filename'])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(item['label'])
        
        return img, torch.tensor(label, dtype=torch.float32)


class CachedTensorDataset(Dataset):
    def __init__(self, tensor_dir: str, label_json: str):
        self.tensor_dir = tensor_dir
        with open(label_json, "r") as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tensor_path = os.path.join(self.tensor_dir, f"{idx}.pt")
        x = torch.load(tensor_path)
        y = float(self.labels[idx])
        return x, torch.tensor(y, dtype=torch.float32)


class TensorInferenceDataset(Dataset):
    def __init__(self, tensor_dir: str, labels):
        self.tensor_dir = tensor_dir
        self.labels = [int(l) for l in labels]
        self.filenames = [f"{i}.pt" for i in range(len(self.labels))]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tensor_path = os.path.join(self.tensor_dir, self.filenames[idx])
        x = torch.load(tensor_path)
        y = float(self.labels[idx])
        fname = self.filenames[idx]
        return x, torch.tensor(y, dtype=torch.float32), fname
