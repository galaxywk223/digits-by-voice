from typing import List, Tuple

import numpy as np


def rms_envelope(y: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    y2 = y ** 2
    n_frames = 1 + max(0, (len(y) - frame_len) // hop_len)
    if n_frames <= 0:
        return np.zeros((0,), dtype=np.float32)
    idx = (np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]).astype(int)
    rms = np.sqrt(np.mean(y2[idx], axis=1) + 1e-12)
    return rms.astype(np.float32)


def split_on_silence(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    min_speech_ms: float = 150.0,
    min_silence_ms: float = 120.0,
    pad_ms: float = 50.0,
) -> List[Tuple[int, int]]:
    """
    Return list of (start, end) sample indices for speech segments.
    Simple energy-based VAD with adaptive threshold.
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)

    env = rms_envelope(y, frame_len, hop_len)
    if env.size == 0:
        return []

    # Adaptive threshold based on statistics
    med = float(np.median(env))
    mx = float(np.max(env))
    # weight controls sensitivity; closer to med for quiet env
    thr = med + 0.25 * (mx - med)

    active = env > thr

    # Merge frames to contiguous segments
    segments = []
    i = 0
    min_speech_frames = max(1, int((min_speech_ms / 1000.0 * sr - frame_len) / hop_len) + 1)
    min_silence_frames = max(1, int(min_silence_ms / 1000.0 * sr / hop_len))

    while i < len(active):
        if not active[i]:
            i += 1
            continue
        # start of speech
        start_f = i
        while i < len(active) and active[i]:
            i += 1
        end_f = i - 1

        # filter too short speech
        if end_f - start_f + 1 < min_speech_frames:
            continue

        # expand to include small gaps shorter than min_silence_frames
        # (robust to quick dropouts)
        j = i
        while j < len(active):
            # look ahead for small silence gap
            gap = 0
            k = j
            while k < len(active) and not active[k] and gap < min_silence_frames:
                gap += 1
                k += 1
            if gap > 0 and gap < min_silence_frames and k < len(active) and active[k]:
                # merge this gap and continue extending speech
                j = k
                while j < len(active) and active[j]:
                    j += 1
                end_f = j - 1
                i = j
            else:
                break

        # convert frame indices to sample indices and pad
        start = max(0, start_f * hop_len - int(sr * pad_ms / 1000.0))
        end = min(len(y), end_f * hop_len + frame_len + int(sr * pad_ms / 1000.0))
        segments.append((start, end))

    # Merge overlapping segments after padding
    merged = []
    for s, e in sorted(segments):
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]

