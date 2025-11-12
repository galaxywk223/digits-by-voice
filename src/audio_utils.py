import os
import sys
import time
import wave
from typing import Tuple

import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_wav(path: str, data: np.ndarray, sr: int) -> None:
    """
    Save mono float32 numpy array (-1..1) to 16-bit PCM WAV.
    """
    _ensure_dir(os.path.dirname(path) or ".")
    data16 = np.clip((data * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(data16.tobytes())


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """
    Read mono or stereo WAV and return mono float32 array (-1..1) and sample rate.
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM supported, got {sampwidth*8}-bit")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels == 2:
        data = data.reshape(-1, 2).mean(axis=1)
    return data, framerate


def record_audio(seconds: float, sr: int = 16000) -> np.ndarray:
    """
    Record mono audio using sounddevice. Returns float32 numpy array (-1..1).
    """
    try:
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError(
            "需要安装 sounddevice 才能录音：pip install sounddevice\n"
            f"导入失败信息：{e}"
        )

    frames = int(seconds * sr)
    audio = sd.rec(frames, samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def record_audio_manual(sr: int = 16000) -> np.ndarray:
    """
    Interactive recording: press Enter to start, press Enter again to stop.
    Uses sounddevice InputStream with a callback to collect audio chunks.
    """
    try:
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError(
            "需要安装 sounddevice 才能录音：pip install sounddevice\n"
            f"导入失败信息：{e}"
        )

    # Caller prints the start prompt; just wait for Enter
    input()
    play_audio(beep(sr))

    chunks = []

    def callback(indata, frames, time_info, status):  # noqa: D401 unused
        if status:
            # 丢弃状态不做打印，避免刷屏
            pass
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=callback):
        try:
            print("录音中… 按回车结束。")
            input()
        except KeyboardInterrupt:
            pass

    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    data = np.concatenate(chunks, axis=0)[:, 0]
    return data.astype(np.float32)


def beep(sr: int = 16000, freq: float = 880.0, dur: float = 0.15) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)


def play_audio(data: np.ndarray, sr: int = 16000) -> None:
    try:
        import sounddevice as sd
    except Exception:
        return  # 如果没有 sounddevice，忽略播放
    sd.play(data, sr)
    sd.wait()


def countdown(seconds: float = 0.8) -> None:
    end = time.time() + seconds
    while True:
        left = end - time.time()
        if left <= 0:
            break
        # 简短的就地更新
        sys.stdout.write(f"\r即将开始录音… {left:0.1f}s ")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r开始！             \n")
    sys.stdout.flush()
