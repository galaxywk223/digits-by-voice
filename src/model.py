import os
from typing import Dict, Optional

import numpy as np

MODEL_PATH = "models/centroids.npz"


def train_centroid_model(X: np.ndarray, y: np.ndarray, Xf: np.ndarray | None = None, yf: np.ndarray | None = None) -> Dict:
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

    # Optional framewise centroids
    frame_centroids = None
    frame_vars = None
    if Xf is not None and yf is not None and Xf.size:
        n_mfcc_f = Xf.shape[1]
        frame_centroids = np.zeros((n_classes, n_mfcc_f), dtype=np.float32)
        frame_vars = np.zeros((n_classes, n_mfcc_f), dtype=np.float32)
        for c in range(n_classes):
            idx = np.where(yf == c)[0]
            if idx.size > 0:
                fc = Xf[idx]
                frame_centroids[c] = fc.mean(axis=0)
                var = fc.var(axis=0) + 1e-4  # variance floor
                frame_vars[c] = var.astype(np.float32)
            else:
                frame_centroids[c] = 0.0
                frame_vars[c] = 1.0
    return {
        "centroids": centroids,
        "mask": mask,
        "frame_centroids": frame_centroids,
        "frame_vars": frame_vars,
    }


def save_model(path: str, model: Dict, params: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {"centroids": model["centroids"], "mask": model["mask"]}
    if model.get("frame_centroids") is not None:
        arrays["frame_centroids"] = model["frame_centroids"]
    if model.get("frame_vars") is not None:
        arrays["frame_vars"] = model["frame_vars"]
    arrays.update({f"param_{k}": v for k, v in params.items()})
    np.savez_compressed(path, **arrays)


def load_model(path: str = MODEL_PATH) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    data = np.load(path, allow_pickle=False)
    centroids = data["centroids"]
    mask = data["mask"]
    frame_centroids = data["frame_centroids"] if "frame_centroids" in data.files else None
    frame_vars = data["frame_vars"] if "frame_vars" in data.files else None
    # Extract params back (prefixed)
    params = {}
    for key in data.files:
        if key.startswith("param_"):
            params[key[len("param_"):]] = data[key].item() if data[key].shape == () else data[key]
    return {"centroids": centroids, "mask": mask, "frame_centroids": frame_centroids, "frame_vars": frame_vars, "params": params}


def predict_digit(feat: np.ndarray, model: Dict) -> int:
    centroids = model["centroids"]
    mask = model["mask"]
    valid = np.where(mask)[0]
    if valid.size == 0:
        raise RuntimeError("模型中没有可用类别，请先录制模板并训练。")
    dists = np.linalg.norm(centroids[valid] - feat[None, :], axis=1)
    return int(valid[np.argmin(dists)])

def predict_frame_class(feat: np.ndarray, model: Dict) -> int:
    """Predict digit for a single frame feature using frame centroids if available."""
    cents = model.get("frame_centroids")
    if cents is None:
        cents = model["centroids"]
    mask = model["mask"]
    valid = np.where(mask)[0]
    # If we have variances, use diagonal Gaussian log-likelihood
    vars_ = model.get("frame_vars")
    if vars_ is not None and vars_ is not False:
        diff = cents[valid] - feat[None, :]
        invvar = 1.0 / (vars_[valid] + 1e-6)
        ll = -0.5 * (np.sum(diff * diff * invvar, axis=1) + np.sum(np.log(vars_[valid] + 1e-6), axis=1))
        return int(valid[np.argmax(ll)])
    # fallback: Euclidean distance
    dists = np.linalg.norm(cents[valid] - feat[None, :], axis=1)
    return int(valid[np.argmin(dists)])

def frame_loglikes(feats: np.ndarray, model: Dict) -> np.ndarray:
    """Return per-frame log-likelihoods matrix of shape (T, C)."""
    cents = model.get("frame_centroids")
    if cents is None:
        cents = model["centroids"]
    vars_ = model.get("frame_vars")
    mask = model["mask"]
    valid = np.where(mask)[0]
    T, D = feats.shape
    C = 10
    ll = np.full((T, C), -1e9, dtype=np.float32)
    if vars_ is not None and vars_ is not False:
        invvar = 1.0 / (vars_[valid] + 1e-6)
        const = -0.5 * np.sum(np.log(vars_[valid] + 1e-6), axis=1)
        for idx, c in enumerate(valid):
            diff = feats - cents[c][None, :]
            ll[:, c] = -0.5 * np.sum(diff * diff * invvar[idx][None, :], axis=1) + const[idx]
    else:
        for c in valid:
            diff = feats - cents[c][None, :]
            ll[:, c] = -np.sum(diff * diff, axis=1)
    return ll
