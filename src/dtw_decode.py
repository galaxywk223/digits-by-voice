from typing import List, Tuple

import numpy as np

from .features import mfcc
from .segment import segment_into_three, extract_main_speech, segment_three_by_peaks
from .dataset import list_wavs
from .audio_utils import read_wav


def _dtw_distance(A: np.ndarray, B: np.ndarray) -> float:
    Ta, D = A.shape
    Tb = B.shape[0]
    if Ta == 0 or Tb == 0:
        return 1e9
    acc = np.full((Ta + 1, Tb + 1), np.inf, dtype=np.float32)
    acc[0, 0] = 0.0
    for i in range(1, Ta + 1):
        a = A[i - 1]
        for j in range(1, Tb + 1):
            b = B[j - 1]
            cost = np.linalg.norm(a - b)
            acc[i, j] = cost + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    path_len = Ta + Tb
    return float(acc[Ta, Tb] / max(1, path_len))


def _utterance_mfcc(y: np.ndarray, sr: int, n_mfcc: int, n_mels: int, frame_ms: float, hop_ms: float) -> np.ndarray:
    y = extract_main_speech(y, sr=sr, frame_ms=frame_ms, hop_ms=hop_ms)
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2
    C = mfcc(y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, frame_len=frame_len, hop_len=hop_len, n_fft=n_fft)
    if C.shape[0] == 0:
        return C
    return C - C.mean(axis=0, keepdims=True)


def build_template_mfccs(data_root: str, sr: int, n_mfcc: int, n_mels: int, frame_ms: float, hop_ms: float):
    templates = {}
    items = list_wavs(data_root)
    for path, label in items:
        y, wav_sr = read_wav(path)
        if wav_sr != sr:
            continue
        C = _utterance_mfcc(y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, frame_ms=frame_ms, hop_ms=hop_ms)
        if C.shape[0] == 0:
            continue
        templates.setdefault(int(label), []).append(C)
    return templates


def classify_by_dtw(seg: np.ndarray, sr: int, templates: dict, n_mfcc: int, n_mels: int, frame_ms: float, hop_ms: float) -> int:
    Cseg = _utterance_mfcc(seg, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, frame_ms=frame_ms, hop_ms=hop_ms)
    if Cseg.shape[0] == 0:
        return 0
    best_c = None
    best_d = 1e9
    for lab, seqs in templates.items():
        for Ctmp in seqs:
            d = _dtw_distance(Cseg, Ctmp)
            if d < best_d:
                best_d = d
                best_c = int(lab)
    return best_c if best_c is not None else 0


def dtw_decode_three(
    y: np.ndarray,
    sr: int,
    data_root: str,
    n_mfcc: int = 13,
    n_mels: int = 26,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    segs = segment_three_by_peaks(y, sr=sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if not segs:
        segs = segment_into_three(y, sr=sr, frame_ms=frame_ms, hop_ms=hop_ms)
    templates = build_template_mfccs(data_root, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, frame_ms=frame_ms, hop_ms=hop_ms)
    digits = []
    for s, e in segs:
        seg = y[s:e]
        c = classify_by_dtw(seg, sr=sr, templates=templates, n_mfcc=n_mfcc, n_mels=n_mels, frame_ms=frame_ms, hop_ms=hop_ms)
        digits.append(c)
    return digits, segs
