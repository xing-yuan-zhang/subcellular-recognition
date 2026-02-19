import json
import os
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import config
from datasets import CachedTensorDataset
from models import build_model


def _build_dataloaders_for_organelle(organelle: str) -> Tuple[DataLoader, DataLoader]:
    train_dir = config.train_cache_dir(organelle)
    index_json = config.train_index_json(organelle)

    dataset = CachedTensorDataset(train_dir, index_json)

    val_size = int(len(dataset) * config.VAL_SPLIT)
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_single_model(
    model_name: str,
    organelle: str,
    pretrained: bool = True,
) -> Dict[str, float]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = _build_dataloaders_for_organelle(organelle)

    model = build_model(model_name, num_outputs=1, pretrained=pretrained)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    best_f1 = 0.0
    patience_counter = 0
    best_epoch = -1

    train_losses = []
    val_accuracies = []
    val_f1s = []

    save_path = config.model_path(model_name, organelle)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(
            train_loader,
            desc=f"[{organelle}][{model_name}] Epoch {epoch + 1}/{config.EPOCHS}",
        ):
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1)  # (B, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels_np = labels.numpy().astype(int)

                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)

                all_labels.extend(labels_np.tolist())
                all_preds.extend(preds.tolist())

        all_labels = np.array(all_labels, dtype=np.int32)
        all_preds = np.array(all_preds, dtype=np.int32)

        val_acc = np.mean(all_preds == all_labels)
        val_f1 = f1_score(all_labels, all_preds)

        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

        print(
            f"[{organelle}][{model_name}] Epoch {epoch + 1}: "
            f"TrainLoss={avg_train_loss:.4f}, ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print("  -> Early stopping")
                break

    metrics = {
        "best_val_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(train_losses[-1]),
        "final_val_acc": float(val_accuracies[-1]),
        "final_val_f1": float(val_f1s[-1]),
    }

    metrics_path = save_path + ".metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{organelle}][{model_name}] Finished training. Best F1={best_f1:.4f} at epoch {best_epoch}")
    return metrics


def train_all_organelle_models(
    model_names=None,
    organelles=None,
    pretrained: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:

    if model_names is None:
        model_names = config.MODEL_NAMES
    if organelles is None:
        organelles = config.ORGANELLES

    results = {}

    for organelle in organelles:
        results[organelle] = {}
        for model_name in model_names:
            print(f"==== Training {model_name} for organelle: {organelle} ====")
            metrics = train_single_model(model_name, organelle, pretrained=pretrained)
            results[organelle][model_name] = metrics

    return results
