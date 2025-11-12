import argparse
import time
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except Exception as e:
    sd = None
    sf = None


def list_input_devices():
    if sd is None:
        print("sounddevice 未安装，无法列出设备")
        return
    print("可用输入设备：")
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            print(f"[{idx}] {dev['name']} (in: {dev['max_input_channels']}, sr: {dev['default_samplerate']:.0f})")


def countdown(n: int = 3):
    for i in range(n, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("开始！")


def record(duration: float, sr: int = 16000, channels: int = 1, device: int | None = None) -> np.ndarray:
    if sd is None:
        raise RuntimeError("未安装 sounddevice/soundfile，无法录音。请先 pip install -r requirements.txt")
    sd.default.samplerate = sr
    sd.default.channels = channels
    if device is not None:
        sd.default.device = device
    frames = int(duration * sr)
    print(f"录音中（{duration:.2f}s, {sr} Hz）...")
    data = sd.rec(frames, dtype="float32")
    sd.wait()
    data = np.squeeze(data)
    return data


def save_wav(path: str | Path, y: np.ndarray, sr: int):
    if sf is None:
        raise RuntimeError("未安装 soundfile，无法保存 WAV 文件。")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(description="录音/设备工具")
    parser.add_argument("--list-devices", action="store_true", help="列出可用输入设备")
    parser.add_argument("--duration", type=float, default=2.0, help="录音时长秒")
    parser.add_argument("--sr", type=int, default=16000, help="采样率")
    parser.add_argument("--device", type=int, default=None, help="输入设备编号（可选）")
    parser.add_argument("--out", type=str, default=None, help="如指定则保存到该 wav 路径")
    args = parser.parse_args()

    if args.__dict__.get("list_devices"):
        list_input_devices()
        return

    countdown(3)
    y = record(duration=args.duration, sr=args.sr, device=args.device)
    if args.out:
        save_wav(args.out, y, args.sr)
        print(f"已保存: {args.out}")
    else:
        print(f"录音完成，长度: {len(y)/args.sr:.2f}s")


if __name__ == "__main__":
    main()

