from __future__ import annotations

from typing import List, Tuple

import numpy as np
import librosa


def non_silent_intervals(y: np.ndarray, sr: int, top_db: int = 35,
                         min_seg_sec: float = 0.18) -> List[Tuple[int, int]]:
    """使用 librosa.effects.split 获取非静音区间，并过滤太短的片段。"""
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
    filtered = []
    min_len = int(min_seg_sec * sr)
    for s, e in intervals:
        if e - s >= min_len:
            filtered.append((int(s), int(e)))
    return filtered


def pick_three_segments(y: np.ndarray, sr: int) -> List[Tuple[int, int]]:
    """尝试获取 3 个语音片段：
    - 逐步调整阈值 top_db，找到 >=3 段
    - 若多于 3 段，取时长最长的 3 段
    - 若仍少于 3 段，返回已有（交由上层决定是否重录）
    返回样本索引区间列表。
    """
    for top_db in (35, 30, 25, 20):
        segs = non_silent_intervals(y, sr, top_db=top_db)
        if len(segs) >= 3:
            # 取最长的 3 段，按起点排序
            segs = sorted(segs, key=lambda ab: (ab[1]-ab[0]), reverse=True)[:3]
            segs = sorted(segs, key=lambda ab: ab[0])
            return segs
    # 如果还是不足 3 段，返回当前能取到的（可能是 0/1/2 段）
    segs = non_silent_intervals(y, sr, top_db=20)
    segs = sorted(segs, key=lambda ab: ab[0])
    return segs

