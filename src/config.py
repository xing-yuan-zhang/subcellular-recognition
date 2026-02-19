import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")

# Expected structure:
#   CACHE_ROOT/
#      focal_adhesion/
#          train/
#              0.pt, 1.pt, ...
#              index_label.json
#          test/
#              0.pt, 1.pt, ...
#              index_label.json

CACHE_ROOT = os.path.join(PROJECT_ROOT, "tensor_cache_by_organelle")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ORGANELLES = [
    "focal_adhesion",
    "mitochondria",
    "ribosome",
    "filopodia",
    "er",
    "nucleus",
]

DEFAULT_ORGANELLE = "focal_adhesion"

# models/{model_name}_{organelle}.pt
def model_path(model_name: str, organelle: str) -> str:
    filename = f"{model_name}_{organelle}.pt"
    return os.path.join(MODEL_DIR, filename)


DEFAULT_MODEL_NAME = "resnet18"  # Allowed: "resnet18", "vgg16", "densenet121", "efficientnet_b0"

MODEL_NAMES = [
    "resnet18",
    "vgg16",
    "densenet121",
    "efficientnet_b0",
]

BATCH_SIZE = 128
INFER_BATCH_SIZE = 64

EPOCHS = 25
LR = 1e-4
PATIENCE = 8
VAL_SPLIT = 0.2
NUM_WORKERS = 4

TILE_SIZE = 256
FULL_SIZE = 2048

# Max misclassified samples to save (optional utility)
MAX_MISCLASSIFIED_SAVE = 50


def train_cache_dir(organelle: str) -> str:
    """Directory containing cached train tensors for a given organelle."""
    return os.path.join(CACHE_ROOT, organelle, "train")


def test_cache_dir(organelle: str) -> str:
    """Directory containing cached test tensors for a given organelle."""
    return os.path.join(CACHE_ROOT, organelle, "test")


def train_index_json(organelle: str) -> str:
    """JSON list of train labels for a given organelle."""
    return os.path.join(train_cache_dir(organelle), "index_label.json")


def test_index_json(organelle: str) -> str:
    """JSON list of test labels for a given organelle."""
    return os.path.join(test_cache_dir(organelle), "index_label.json")
