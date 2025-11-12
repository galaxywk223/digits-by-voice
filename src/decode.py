from typing import List, Tuple

import numpy as np

from .features import mfcc, mfcc_mean_feature
from .segment import rms_envelope, segment_into_three, extract_main_speech, segment_three_by_peaks
from .model import predict_frame_class, predict_digit, frame_loglikes


def framewise_decode_three(
    y: np.ndarray,
    sr: int,
    model,
    n_mfcc: int = 13,
    n_mels: int = 26,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    min_run_ms: float = 90.0,
    smooth_width: int = 5,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Decode three digits by per-frame classification + run-length grouping.
    Returns (digits, segments(sample index pairs)).
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2

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
        return [], []

    env = rms_envelope(y, frame_len, hop_len)
    env = env[: coeffs.shape[0]]
    # energy threshold to mask silence frames
    med = float(np.median(env)) if env.size else 0.0
    mx = float(np.max(env)) if env.size else 0.0
    thr = med + 0.18 * (mx - med) if env.size else 0.0
    active = env > thr

    # cepstral mean normalization per utterance (use active frames mean)
    if np.any(active):
        cmn = coeffs[active].mean(axis=0, keepdims=True)
    else:
        cmn = coeffs.mean(axis=0, keepdims=True)
    coeffs = coeffs - cmn

    # Predict class per frame
    preds = []
    for t in range(coeffs.shape[0]):
        if not active[t]:
            preds.append(-1)
        else:
            preds.append(predict_frame_class(coeffs[t], model))
    preds = np.array(preds, dtype=int)

    # Smooth with median filter (ignore -1 by temporary replacement)
    if smooth_width >= 3:
        arr = preds.copy()
        mask = arr == -1
        fill = -100
        arr[mask] = fill
        k = smooth_width
        pad = k // 2
        pad_left = np.full((pad,), fill, dtype=int)
        pad_right = np.full((pad,), fill, dtype=int)
        arr_pad = np.concatenate([pad_left, arr, pad_right])
        sm = []
        for i in range(len(arr)):
            win = arr_pad[i : i + k]
            # exclude fill when computing median
            wv = win[win != fill]
            if wv.size == 0:
                sm.append(-1)
            else:
                sm.append(int(np.median(wv)))
        preds = np.array(sm, dtype=int)

    # Run-length grouping ignoring -1
    runs: List[Tuple[int, int, int]] = []  # (start_frame, end_frame_exclusive, label)
    i = 0
    while i < len(preds):
        if preds[i] == -1:
            i += 1
            continue
        j = i + 1
        while j < len(preds) and preds[j] == preds[i]:
            j += 1
        runs.append((i, j, preds[i]))
        i = j

    min_run_frames = max(1, int(min_run_ms / 1000.0 * sr / hop_len))
    runs = [r for r in runs if (r[1] - r[0]) >= min_run_frames]
    if not runs:
        return [], []

    # If >3 runs, keep three strongest by duration*avg_energy
    def run_score(r):
        s, e, lab = r
        dur = e - s
        avg_e = float(np.mean(env[s:e])) if e > s else 0.0
        return dur * avg_e

    if len(runs) > 3:
        runs = sorted(runs, key=run_score, reverse=True)[:3]
        runs = sorted(runs, key=lambda x: x[0])

    # If insufficient diversity (e.g., runs < 3), fallback to energy-based segmentation
    if len(runs) < 3 or len({r[2] for r in runs}) < 2:
        segs = segment_three_by_peaks(y, sr=sr, frame_ms=frame_ms, hop_ms=hop_ms)
        if not segs:
            segs = segment_into_three(y, sr=sr, frame_ms=frame_ms, hop_ms=hop_ms)
        digits = []
        for s, e in segs:
            seg = y[s:e]
            seg = extract_main_speech(seg, sr=sr, frame_ms=frame_ms, hop_ms=hop_ms)
            feat = mfcc_mean_feature(seg, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, frame_ms=frame_ms, hop_ms=hop_ms)
            digits.append(predict_digit(feat, model))
        return digits, segs

    # Else convert top three runs by score
    runs = runs[:3]
    digits = [r[2] for r in runs]
    segs = [(r[0] * hop_len, r[1] * hop_len + frame_len) for r in runs]
    return digits, segs


def dp_decode_three(
    y: np.ndarray,
    sr: int,
    model,
    n_mfcc: int = 13,
    n_mels: int = 26,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    min_run_ms: float = 90.0,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Dynamic programming segmentation on per-frame Gaussian log-likelihoods.
    Find boundaries t1,t2 maximizing segment sums for unknown digits.
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2
    coeffs = mfcc(y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, frame_len=frame_len, hop_len=hop_len, n_fft=n_fft)
    if coeffs.shape[0] == 0:
        return [], []
    # Per-utterance CMN
    coeffs = coeffs - coeffs.mean(axis=0, keepdims=True)
    T = coeffs.shape[0]

    # Energy gating to trim leading/trailing silence
    env = rms_envelope(y, frame_len, hop_len)
    env = env[:T]
    if env.size:
        med = float(np.median(env)); mx = float(np.max(env))
        thr = med + 0.18 * (mx - med)
        active = env > thr
        if np.any(active):
            idx = np.where(active)[0]
            lo, hi = int(idx[0]), int(idx[-1])
            lo = max(0, lo - 2); hi = min(T - 1, hi + 2)
            coeffs = coeffs[lo : hi + 1]
            T = coeffs.shape[0]

    # Log-likelihood per frame and digit
    LL = frame_loglikes(coeffs, model)  # (T, C)

    m = max(2, int(min_run_ms / 1000.0 * sr / hop_len))
    if 3 * m > T:
        m = max(1, T // 5)

    # Prefix sums for quick segment score
    P = np.cumsum(LL, axis=0)  # (T, C)

    def segsum(d: int, a: int, b: int) -> float:
        if a == 0:
            return float(P[b, d])
        return float(P[b, d] - P[a - 1, d])

    # DP for two boundaries
    best1 = np.full((T,), -1e18, dtype=np.float64)
    arg1_d = np.full((T,), -1, dtype=int)
    for t in range(m - 1, T):
        # best first segment ending at t
        vals = P[t, :]
        d = int(np.argmax(vals))
        best1[t] = float(vals[d])
        arg1_d[t] = d

    best12 = np.full((T,), -1e18, dtype=np.float64)
    arg12_t1 = np.full((T,), -1, dtype=int)
    arg12_d2 = np.full((T,), -1, dtype=int)
    for t2 in range(2 * m - 1, T - m):
        # choose t1 in [m-1 .. t2-m]
        best_val = -1e18
        best_t1 = -1
        best_d2 = -1
        for t1 in range(m - 1, t2 - m + 1):
            # best d2 for segment (t1+1..t2)
            vals = P[t2, :] - P[t1, :]
            d2 = int(np.argmax(vals))
            val = best1[t1] + float(vals[d2])
            if val > best_val:
                best_val = val
                best_t1 = t1
                best_d2 = d2
        best12[t2] = best_val
        arg12_t1[t2] = best_t1
        arg12_d2[t2] = best_d2

    # Final boundary t3 = T-1
    best = -1e18
    bt1 = bt2 = -1
    bd1 = bd2 = bd3 = -1
    t3 = T - 1
    for t2 in range(2 * m - 1, T - m + 1):
        vals = P[t3, :] - P[t2, :]
        d3 = int(np.argmax(vals))
        val = best12[t2] + float(vals[d3])
        if val > best:
            best = val
            bt2 = t2
            bt1 = arg12_t1[t2]
            bd2 = arg12_d2[t2]
            bd3 = d3
    if bt1 < 0:
        return [], []
    bd1 = arg1_d[bt1]

    # Convert frame indices to sample segments
    segs_f = [(0, bt1), (bt1 + 1, bt2), (bt2 + 1, t3)]
    segs = [(s * hop_len, e * hop_len + frame_len) for s, e in segs_f]
    digits = [bd1, bd2, bd3]
    return digits, segs
