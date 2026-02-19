# Cross organelle generalization with CNNs

Train and maintain a small suite of lab-internal pretrained CNN models for fluorescence microscopy, used to automatically recognize subcellular organelles from image patches. This repository supports routine dataset curation + training + evaluation for organelle-specific binary classifiers, and tests cross-organelle generalization to understand robustness and failure modes.

<p align="center">
  <img width="450" height="175" alt="Screenshot 2025-12-11 133441" src="https://github.com/user-attachments/assets/b613e979-c109-4c2d-833f-fe0f534d59a6" />
</p>

## Description
- Models: organelle-specific binary CNN classifiers (ResNet18, VGG16, DenseNet121, EfficientNet-B0)
- Input: 128×128 fluorescence microscopy patches
- Tasks:
  - Train per-organelle classifiers and save reusable pretrained checkpoints for downstream lab workflows
  - Evaluate cross-organelle transfer/generalization and qualitative error inspection including hard negatives, bleed-through and low SNR

## Setup

### Dependencies
- Python 3.9+
- torchvision
- numpy / opencv-python / scikit-learn (for I/O + metrics)

### Quickstart
`main.py` is the entry point and a typical workflow is:

```bash
python main.py --mode train --organelle mito --arch resnet18 --patch_size 128 --data_root data/ --save_ckpt ckpts/mito_resnet18.pt
python main.py --mode eval  --ckpt ckpts/mito_resnet18.pt --test_organelle golgi --data_root data/ --n_vis 64
python main.py --mode sweep --train_organelle mito --test_organelles mito golgi er nucleus --archs resnet18 vgg16 densenet121 efficientnet_b0 --out results/sweep.json
```

## Notes
- This repo is intended for internal research and model maintenance in our lab, not as a polished benchmark.

