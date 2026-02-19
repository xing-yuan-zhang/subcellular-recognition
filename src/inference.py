import json
import os
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import torch
from torch.utils.data import DataLoader

import config
from datasets import TensorInferenceDataset
from models import build_model


def _load_labels(label_json_path: str) -> List[int]:
    with open(label_json_path, "r") as f:
        labels = json.load(f)
    return [int(l) for l in labels]


def evaluate_on_organelle(
    model_name: str,
    train_organelle: str,
    test_organelle: str,
) -> Dict[str, float]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_name, num_outputs=1, pretrained=False)
    model_path = config.model_path(model_name, train_organelle)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    test_dir = config.test_cache_dir(test_organelle)
    index_json = config.test_index_json(test_organelle)
    labels = _load_labels(index_json)
    dataset = TensorInferenceDataset(test_dir, labels)

    loader = DataLoader(
        dataset,
        batch_size=config.INFER_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for x, y, _fname in loader:
            x = x.to(device)
            y_true_np = y.numpy().astype(int)

            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            y_true.extend(y_true_np.tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    y_prob = np.array(y_prob, dtype=np.float32)

    acc = float(np.mean(y_pred == y_true))
    f1 = float(
        (2 * ( (y_pred & y_true).sum() )) /
        ( (y_pred.sum() + y_true.sum()) + 1e-8 )
    )

    from sklearn.metrics import f1_score
    f1 = float(f1_score(y_true, y_pred))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    print(
        f"[EVAL] model={model_name}, train_on={train_organelle}, test_on={test_organelle} "
        f"-> Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
    )
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "cm": cm.tolist(),
    }


def cross_eval_matrix(
    model_name: str,
    organelles=None,
) -> Dict[str, Dict[str, Dict[str, float]]]:

    if organelles is None:
        organelles = config.ORGANELLES

    results = {}
    for train_org in organelles:
        results[train_org] = {}
        for test_org in organelles:
            metrics = evaluate_on_organelle(
                model_name=model_name,
                train_organelle=train_org,
                test_organelle=test_org,
            )
            results[train_org][test_org] = metrics

    return results
