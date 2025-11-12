import os
from typing import Dict, List, Tuple

import numpy as np

from .audio_utils import read_wav
from .features import mfcc_mean_feature, mfcc
from .segment import extract_main_speech, rms_envelope


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
        # Trim leading/trailing silence to robustify templates
        y = extract_main_speech(y, sr=sr)
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


def load_framewise_features(
    sr: int = 16000,
    n_mfcc: int = 13,
    n_mels: int = 26,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    data_root: str = "data/raw",
):
    """
    Return (XF, yF):
    - XF shape (M, n_mfcc) per-frame MFCCs concatenated across all files
    - yF shape (M,) corresponding digit label per frame
    """
    items = list_wavs(data_root)
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2
    feats = []
    labels = []
    for path, label in items:
        y, this_sr = read_wav(path)
        if this_sr != sr:
            raise ValueError(f"采样率不一致：{path} 为 {this_sr}，期望 {sr}")
        # trim leading/trailing silence per utterance
        y = extract_main_speech(y, sr=sr)
        coeffs = mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            frame_len=frame_len,
            hop_len=hop_len,
            n_fft=n_fft,
        )
        if coeffs.shape[0] == 0:
            continue
        # keep only speech frames by energy threshold
        env = rms_envelope(y, frame_len, hop_len)
        env = env[: coeffs.shape[0]]
        med = float(np.median(env)) if env.size else 0.0
        mx = float(np.max(env)) if env.size else 0.0
        thr = med + 0.20 * (mx - med) if env.size else 0.0
        active = env > thr if env.size else np.ones((coeffs.shape[0],), dtype=bool)
        if not np.any(active):
            # fallback: take top-60% frames by energy
            k = max(1, int(round(0.6 * len(env))))
            idx = np.argsort(env)[-k:]
            active = np.zeros_like(env, dtype=bool)
            active[idx] = True
        # per-utterance CMN on active frames
        cmn = coeffs[active].mean(axis=0, keepdims=True)
        coeffs = coeffs - cmn
        feats.append(coeffs[active])
        labels.append(np.full((int(np.sum(active)),), label, dtype=np.int64))
    if not feats:
        return np.zeros((0, n_mfcc), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    XF = np.concatenate(feats, axis=0).astype(np.float32)
    yF = np.concatenate(labels, axis=0)
    return XF, yF
