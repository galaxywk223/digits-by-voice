import os
from typing import Dict, List, Tuple

import numpy as np

from .audio_utils import read_wav
from .features import mfcc_mean_feature


DIGITS = [str(i) for i in range(10)]


def list_wavs(data_root: str = "data/raw") -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    for d in DIGITS:
        cls_dir = os.path.join(data_root, d)
        if not os.path.isdir(cls_dir):
            continue
        for name in sorted(os.listdir(cls_dir)):
            if not name.lower().endswith('.wav'):
                continue
            items.append((os.path.join(cls_dir, name), int(d)))
    return items


def load_features(
    sr: int = 16000,
    n_mfcc: int = 13,
    n_mels: int = 26,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    data_root: str = "data/raw",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (X, y): X shape (N, n_mfcc), y shape (N,)
    """
    items = list_wavs(data_root)
    feats: List[np.ndarray] = []
    labels: List[int] = []
    for path, label in items:
        y, this_sr = read_wav(path)
        if this_sr != sr:
            raise ValueError(f"采样率不一致：{path} 为 {this_sr}，期望 {sr}")
        feat = mfcc_mean_feature(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
        )
        feats.append(feat)
        labels.append(label)
    if not feats:
        return np.zeros((0, n_mfcc), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.stack(feats, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y

