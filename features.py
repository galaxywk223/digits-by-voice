from __future__ import annotations

import numpy as np
import librosa


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """提取一段语音的特征向量。

    - 预加重
    - MFCC (13) + 一阶/二阶差分
    - 对每一组做 time 维度的 mean/std 聚合
    输出形状: (13*2*3,) = 78 维
    """
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    # 去 DC、归一化
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    # 预加重
    y = librosa.effects.preemphasis(y)

    # 窗长/步长（约 32ms / 10ms）
    n_fft = 512
    hop_length = 160
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    def stats(feat: np.ndarray) -> np.ndarray:
        m = np.mean(feat, axis=1)
        s = np.std(feat, axis=1) + 1e-10
        return np.concatenate([m, s], axis=0)

    feat_vec = np.concatenate([stats(mfcc), stats(d1), stats(d2)], axis=0)
    return feat_vec.astype(np.float32)

