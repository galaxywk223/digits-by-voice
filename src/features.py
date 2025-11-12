from typing import Tuple

import numpy as np


def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_fft: int, n_mels: int, sr: int, fmin: float, fmax: float) -> np.ndarray:
    """
    Create Mel filterbank matrix of shape (n_mels, 1 + n_fft//2)
    """
    # FFT bin frequencies
    freqs = np.linspace(0, sr / 2, 1 + n_fft // 2)

    # mel points
    mels = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz = mel_to_hz(mels)

    fb = np.zeros((n_mels, freqs.shape[0]), dtype=np.float32)
    for i in range(1, n_mels + 1):
        f_left, f_center, f_right = hz[i - 1], hz[i], hz[i + 1]
        # rising slope
        left_slope = (freqs - f_left) / (f_center - f_left + 1e-9)
        # falling slope
        right_slope = (f_right - freqs) / (f_right - f_center + 1e-9)
        fb[i - 1] = np.maximum(0.0, np.minimum(left_slope, right_slope))
    # Slaney-style energy normalization
    enorm = 2.0 / (hz[2 : n_mels + 2] - hz[: n_mels])
    fb *= enorm[:, np.newaxis]
    return fb


def dct_type_2(n_mfcc: int, n_mels: int) -> np.ndarray:
    """
    DCT-II transform matrix of shape (n_mfcc, n_mels)
    """
    n = np.arange(n_mels)
    k = np.arange(n_mfcc)[:, None]
    mat = np.cos((np.pi / n_mels) * (n + 0.5) * k)
    mat[0, :] *= 1.0 / np.sqrt(2.0)
    return mat * np.sqrt(2.0 / n_mels)


def frame_signal(y: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    n_frames = 1 + max(0, (len(y) - frame_len) // hop_len)
    if n_frames <= 0:
        return np.zeros((0, frame_len), dtype=y.dtype)
    idx = (np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]).astype(int)
    return y[idx]


def mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_mels: int = 26,
    frame_len: int = 400,  # 25 ms at 16k
    hop_len: int = 160,    # 10 ms at 16k
    n_fft: int = 512,
    pre_emph: float = 0.97,
    fmin: float = 20.0,
    fmax: float = None,
) -> np.ndarray:
    """
    Compute MFCCs per frame. Returns array shape (T, n_mfcc)
    Implementation uses only numpy.
    """
    if fmax is None:
        fmax = sr / 2

    # pre-emphasis
    y = np.append(y[0], y[1:] - pre_emph * y[:-1])

    # framing + window
    frames = frame_signal(y, frame_len, hop_len)
    if frames.shape[0] == 0:
        return np.zeros((0, n_mfcc), dtype=np.float32)
    window = np.hamming(frame_len).astype(np.float32)
    frames = frames * window[None, :]

    # FFT power spectrum
    # zero-pad to n_fft
    if frame_len < n_fft:
        pad = np.zeros((frames.shape[0], n_fft - frame_len), dtype=frames.dtype)
        frames_padded = np.concatenate([frames, pad], axis=1)
    else:
        frames_padded = frames[:, :n_fft]
    spec = np.fft.rfft(frames_padded, n=n_fft, axis=1)
    power = (np.abs(spec) ** 2) / n_fft

    # Mel filterbank
    fb = mel_filterbank(n_fft=n_fft, n_mels=n_mels, sr=sr, fmin=fmin, fmax=fmax)
    mel_energies = np.maximum(1e-10, power @ fb.T)

    # log mel
    log_mel = np.log(mel_energies)

    # DCT-II to MFCC
    dct = dct_type_2(n_mfcc=n_mfcc, n_mels=n_mels)
    mfccs = log_mel @ dct.T
    return mfccs.astype(np.float32)


def mfcc_mean_feature(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_mels: int = 26,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
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
        return np.zeros((n_mfcc,), dtype=np.float32)
    return coeffs.mean(axis=0)

