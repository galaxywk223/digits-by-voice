import argparse
from pathlib import Path

import joblib
import numpy as np
import librosa

from audio_io import countdown, record
from features import extract_features
from segment import pick_three_segments


def predict_digit_segments(model, y: np.ndarray, sr: int):
    segs = pick_three_segments(y, sr)
    if len(segs) < 3:
        print(f"检测到的语音段少于 3 段（{len(segs)} 段）。请尝试更明显的停顿或加大音量后重试。")
        return None, None

    digits = []
    probs = []
    for (s, e) in segs[:3]:
        seg = y[s:e]
        feat = extract_features(seg, sr)
        prob = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(feat.reshape(1, -1))[0]
            digit = int(np.argmax(p))
            prob = float(np.max(p))
        else:
            digit = int(model.predict(feat.reshape(1, -1))[0])
        digits.append(digit)
        probs.append(prob)
    return digits, probs


def main():
    parser = argparse.ArgumentParser(description="连续三个数字识别")
    parser.add_argument("--model", type=str, default="models/digit_clf.joblib", help="模型路径")
    parser.add_argument("--duration", type=float, default=6.0, help="录音总时长(秒)")
    parser.add_argument("--sr", type=int, default=16000, help="采样率")
    parser.add_argument("--device", type=int, default=None, help="输入设备编号（可选）")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"未找到模型: {model_path}，请先运行 train_model.py 训练。")

    print(f"加载模型: {model_path}")
    model = joblib.load(model_path)

    print("请准备在倒计时后连续说出三个数字（中间自然停顿）")
    countdown(3)
    y = record(duration=args.duration, sr=args.sr, device=args.device)

    digits, probs = predict_digit_segments(model, y, args.sr)
    if digits is None:
        return
    result = ''.join(str(d) for d in digits)
    print(f"识别结果: {result}")
    if probs and all(p is not None for p in probs):
        print("置信度: " + ", ".join(f"{p:.2f}" for p in probs))


if __name__ == "__main__":
    main()

