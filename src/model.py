import os
from typing import Dict, Optional

import numpy as np

MODEL_PATH = "models/centroids.npz"


def train_centroid_model(X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Compute class-wise centroids for labels 0..9.
    Returns a dict with centroids, mask and params placeholder.
    """
    n_classes = 10
    n_mfcc = X.shape[1] if X.size else 13
    centroids = np.zeros((n_classes, n_mfcc), dtype=np.float32)
    mask = np.zeros((n_classes,), dtype=bool)
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        if idx.size > 0:
            centroids[c] = X[idx].mean(axis=0)
            mask[c] = True
    return {
        "centroids": centroids,
        "mask": mask,
    }


def save_model(path: str, model: Dict, params: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, centroids=model["centroids"], mask=model["mask"], **{f"param_{k}": v for k, v in params.items()})


def load_model(path: str = MODEL_PATH) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    data = np.load(path, allow_pickle=False)
    centroids = data["centroids"]
    mask = data["mask"]
    # Extract params back (prefixed)
    params = {}
    for key in data.files:
        if key.startswith("param_"):
            params[key[len("param_"):]] = data[key].item() if data[key].shape == () else data[key]
    return {"centroids": centroids, "mask": mask, "params": params}


def predict_digit(feat: np.ndarray, model: Dict) -> int:
    centroids = model["centroids"]
    mask = model["mask"]
    valid = np.where(mask)[0]
    if valid.size == 0:
        raise RuntimeError("模型中没有可用类别，请先录制模板并训练。")
    dists = np.linalg.norm(centroids[valid] - feat[None, :], axis=1)
    return int(valid[np.argmin(dists)])

