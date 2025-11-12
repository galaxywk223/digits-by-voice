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
    thr_weight: float = 0.25,
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
    # Optional light smoothing to reduce spurious spikes
    if env.size >= 3:
        kernel = np.ones(3, dtype=np.float32) / 3.0
        env = np.convolve(env, kernel, mode="same").astype(np.float32)
    med = float(np.median(env))
    mx = float(np.max(env))
    thr_weight = float(np.clip(thr_weight, 0.0, 1.0))
    # weight controls sensitivity; closer to med for higher sensitivity
    thr = med + thr_weight * (mx - med)

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


def force_split_into_n(
    y: np.ndarray,
    sr: int,
    n: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> List[Tuple[int, int]]:
    """
    Fallback split: divide the utterance into N chunks by searching
    for local minima of the RMS envelope near equally spaced cut points.
    """
    if n <= 1 or y.size == 0:
        return [(0, len(y))]
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    env = rms_envelope(y, frame_len, hop_len)
    if env.size == 0:
        step = len(y) // n
        return [(i * step, (i + 1) * step if i < n - 1 else len(y)) for i in range(n)]
    # smooth envelope
    if env.size >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        env = np.convolve(env, kernel, mode="same").astype(np.float32)
    # default cut positions at equal frames
    total_frames = env.shape[0]
    cut_frames = []
    for k in range(1, n):
        target = int(round(k * total_frames / n))
        w = max(2, total_frames // 20)  # search window +-w/2
        lo = max(1, target - w)
        hi = min(total_frames - 2, target + w)
        if lo >= hi:
            cut = target
        else:
            local = env[lo:hi]
            cut = lo + int(np.argmin(local))
        cut_frames.append(cut)
    cut_frames = sorted(set(int(c) for c in cut_frames))
    # convert to samples
    samples = [0] + [int(cf * hop_len) for cf in cut_frames] + [len(y)]
    segs: List[Tuple[int, int]] = []
    for i in range(n):
        s = max(0, samples[i])
        e = min(len(y), samples[i + 1])
        if e - s > int(0.05 * sr):  # keep segments longer than 50ms
            segs.append((s, e))
    # if pruning reduced count, pad by merging or adjusting boundaries
    if len(segs) < n:
        # fallback to equal pieces
        step = len(y) // n
        segs = [(i * step, (i + 1) * step if i < n - 1 else len(y)) for i in range(n)]
    return segs


def spectral_flux(y: np.ndarray, frame_len: int, hop_len: int, n_fft: int) -> np.ndarray:
    """Compute simple spectral flux over frames."""
    n_frames = 1 + max(0, (len(y) - frame_len) // hop_len)
    if n_frames <= 1:
        return np.zeros((0,), dtype=np.float32)
    idx = (np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]).astype(int)
    frames = y[idx] * np.hamming(frame_len)[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    mag = np.abs(spec).astype(np.float32)
    diff = np.maximum(0.0, mag[1:] - mag[:-1])
    flux = np.sum(diff, axis=1)
    # pad to length n_frames
    flux = np.concatenate([flux[:1], flux], axis=0)
    return flux


def segment_three_by_peaks(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    pad_ms: float = 25.0,
    min_len_ms: float = 100.0,
) -> List[Tuple[int, int]]:
    """
    Segment into exactly 3 by finding 3 largest energy peaks with distance constraint,
    then cut at local minima between adjacent peaks.
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2
    env = rms_envelope(y, frame_len, hop_len)
    if env.size < 5:
        return []

    # smooth env
    k = max(3, int(round(25.0 / hop_ms)))  # ~250ms window
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k,), dtype=np.float32) / k
    env_s = np.convolve(env, kernel, mode="same")

    # find local maxima
    peaks = []
    for t in range(1, len(env_s) - 1):
        if env_s[t] > env_s[t - 1] and env_s[t] >= env_s[t + 1]:
            peaks.append((env_s[t], t))
    if not peaks:
        return []
    peaks.sort(key=lambda x: x[0], reverse=True)
    min_dist = max(2, int(round(180.0 / hop_ms)))  # ~180ms separation
    selected = []
    for val, t in peaks:
        if all(abs(t - s) >= min_dist for s in selected):
            selected.append(t)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        return []
    selected.sort()
    p1, p2, p3 = selected

    # cut at minima between peaks
    def local_min(lo: int, hi: int) -> int:
        lo = max(lo, 1)
        hi = min(hi, len(env_s) - 2)
        if lo >= hi:
            return (lo + hi) // 2
        seg = env_s[lo:hi + 1]
        return lo + int(np.argmin(seg))

    c1 = local_min(p1, p2)
    c2 = local_min(p2, p3)

    # enforce minimal length
    min_len_f = max(1, int(round(min_len_ms / hop_ms)))
    if c1 < min_len_f:
        c1 = min_len_f
    if c2 - c1 < min_len_f:
        c2 = min(len(env_s) - 2, c1 + min_len_f)
    if (len(env_s) - 1) - c2 < min_len_f:
        c2 = max(c1 + min_len_f, (len(env_s) - 1) - min_len_f)

    pad = int(sr * pad_ms / 1000.0)
    s1 = max(0, c1 * hop_len - pad)
    e1 = min(len(y), c1 * hop_len + frame_len + pad)
    s2 = max(0, c2 * hop_len - pad)
    e2 = min(len(y), c2 * hop_len + frame_len + pad)

    segs = [(0, e1), (s2, e2), (e2, len(y))]
    # Refine to avoid overlaps and zero-length
    segs = [(max(0, s), max(s + 1, min(len(y), e))) for s, e in segs]
    # Ensure order and non-overlap by adjusting middle boundaries
    segs = [(segs[0][0], min(segs[0][1], segs[1][0])), (segs[1][0], min(segs[1][1], segs[2][0])), (segs[2][0], segs[2][1])]
    # Filter too short segments
    segs = [seg for seg in segs if (seg[1] - seg[0]) >= int(sr * (min_len_ms / 1000.0))]
    if len(segs) != 3:
        return []
    return segs


def _merge_to_target(segs: List[Tuple[int, int]], target: int) -> List[Tuple[int, int]]:
    """Iteratively merge nearest segments until reaching target count."""
    segs = sorted(segs)
    if len(segs) <= target:
        return segs
    while len(segs) > target:
        # compute gaps between consecutive segments
        gaps = [(segs[i + 1][0] - segs[i][1], i) for i in range(len(segs) - 1)]
        if not gaps:
            break
        # find smallest gap and merge those two
        _, idx = min(gaps, key=lambda t: t[0])
        a = segs[idx]
        b = segs[idx + 1]
        merged = (a[0], b[1])
        segs = segs[:idx] + [merged] + segs[idx + 2 :]
    return segs


def segment_into_three(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    min_speech_ms: float = 120.0,
    min_silence_ms: float = 100.0,
    pad_ms: float = 50.0,
) -> List[Tuple[int, int]]:
    """
    Robustly segment an utterance with three digits into exactly three segments.
    Strategy:
    - Try a set of thresholds to find 3 speech regions (energy-based VAD).
    - If more than 3 segments, iteratively merge nearest neighbors until 3.
    - If fewer than 3, split the utterance into 3 by finding local minima.
    """
    # candidate threshold weights from sensitive to conservative and back
    weights = [0.20, 0.25, 0.18, 0.30, 0.15, 0.35]
    best: List[Tuple[int, int]] = []
    for w in weights:
        segs = split_on_silence(
            y,
            sr=sr,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            pad_ms=pad_ms,
            thr_weight=w,
        )
        # filter extremely short segments
        segs = [s for s in segs if (s[1] - s[0]) >= int(0.10 * sr)]
        if len(segs) == 3:
            return segs
        if len(segs) > len(best):
            best = segs

    segs = best
    if len(segs) > 3:
        segs = _merge_to_target(segs, 3)
        return segs

    # Not enough segments: derive cut points by envelope minima
    return force_split_into_n(y, sr=sr, n=3, frame_ms=frame_ms, hop_ms=hop_ms)


def extract_main_speech(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    min_speech_ms: float = 80.0,
    min_silence_ms: float = 80.0,
    pad_ms: float = 20.0,
    thr_weight: float = 0.2,
) -> np.ndarray:
    """
    Trim leading/trailing silence by selecting the highest-energy speech segment.
    Returns the trimmed signal; if no segment found, returns original.
    """
    segs = split_on_silence(
        y,
        sr=sr,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        pad_ms=pad_ms,
        thr_weight=thr_weight,
    )
    if not segs:
        return y
    # choose segment with max energy
    def energy(seg):
        s, e = seg
        return float(np.sum(y[s:e] ** 2))
    s, e = max(segs, key=energy)
    return y[s:e]
