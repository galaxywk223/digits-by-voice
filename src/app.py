import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np

from .audio_utils import beep, countdown, play_audio, read_wav, record_audio, write_wav
from .dataset import load_features
from .features import mfcc_mean_feature
from .model import MODEL_PATH, load_model, predict_digit, save_model, train_centroid_model
from .segment import split_on_silence


def cmd_record_templates(args: argparse.Namespace) -> None:
    sr = args.sr
    count = args.count
    dur = args.duration
    out_root = os.path.join(args.data_root, "raw")
    digits = [str(i) for i in range(10)]

    print(f"采样率: {sr} Hz, 每条时长: {dur}s，每个数字录制: {count} 条")
    for d in digits:
        ddir = os.path.join(out_root, d)
        os.makedirs(ddir, exist_ok=True)
        print(f"\n请读数字：{d}")
        for i in range(count):
            input("按回车开始倒计时…")
            play_audio(beep(sr))
            countdown(0.8)
            audio = record_audio(dur, sr=sr)
            ts = int(time.time() * 1000)
            path = os.path.join(ddir, f"sample_{ts}_{i+1}.wav")
            write_wav(path, audio, sr)
            print(f"已保存: {path}")
    print("\n模板录制完成！")


def cmd_train(args: argparse.Namespace) -> None:
    X, y = load_features(
        sr=args.sr,
        n_mfcc=args.n_mfcc,
        n_mels=args.n_mels,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        data_root=args.data_root,
    )
    if X.shape[0] == 0:
        print("未发现样本，请先运行 record-templates。")
        sys.exit(1)
    model = train_centroid_model(X, y)
    params = dict(sr=args.sr, n_mfcc=args.n_mfcc, n_mels=args.n_mels, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    save_model(args.model_path, model, params)
    print(f"模型已保存：{args.model_path}")


def pick_three_segments(segs: List[Tuple[int, int]], y: np.ndarray) -> List[Tuple[int, int]]:
    if len(segs) <= 3:
        return segs
    # 选择能量较高的前三段
    def energy(seg):
        s, e = seg
        return float(np.sum(y[s:e] ** 2))
    ranked = sorted(segs, key=energy, reverse=True)
    # 保持时间顺序输出
    top = sorted(ranked[:3], key=lambda x: x[0])
    return top


def cmd_recognize_3(args: argparse.Namespace) -> None:
    model = load_model(args.model_path)
    if model is None:
        print("未找到模型，请先运行 train 训练。")
        sys.exit(1)

    sr = int(model["params"].get("sr", args.sr))
    n_mfcc = int(model["params"].get("n_mfcc", args.n_mfcc))
    n_mels = int(model["params"].get("n_mels", args.n_mels))
    frame_ms = float(model["params"].get("frame_ms", args.frame_ms))
    hop_ms = float(model["params"].get("hop_ms", args.hop_ms))

    print(f"请在 {args.max_seconds}s 内连续说三个数字（中间留短暂停顿）…")
    play_audio(beep(sr, dur=0.2))
    audio = record_audio(args.max_seconds, sr=sr)

    segs = split_on_silence(audio, sr=sr)
    segs = [seg for seg in segs if (seg[1] - seg[0]) > int(sr * 0.12)]  # 丢弃过短片段
    segs = pick_three_segments(segs, audio)

    if len(segs) != 3:
        print(f"未能检测到三段语音（检测到 {len(segs)} 段），请重新尝试并增加停顿或提高音量。")
        # 打印调试信息
        for i, (s, e) in enumerate(segs):
            print(f"段{i+1}: {s/sr:.2f}s - {e/sr:.2f}s, 长度={(e-s)/sr:.2f}s")
        sys.exit(2)

    digits = []
    for i, (s, e) in enumerate(segs):
        seg = audio[s:e]
        feat = mfcc_mean_feature(seg, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, frame_ms=frame_ms, hop_ms=hop_ms)
        pred = predict_digit(feat, model)
        digits.append(pred)
    print("识别结果：", " ".join(str(d) for d in digits))


def cmd_record_once(args: argparse.Namespace) -> None:
    print(f"试录 {args.duration}s …")
    play_audio(beep(args.sr))
    countdown(0.8)
    audio = record_audio(args.duration, sr=args.sr)
    os.makedirs("tmp", exist_ok=True)
    out = os.path.join("tmp", "test.wav")
    write_wav(out, audio, args.sr)
    print(f"已保存：{out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="音频数字识别（模板+三位连读）")
    sub = p.add_subparsers(dest="cmd", required=True)

    # record-templates
    r = sub.add_parser("record-templates", help="录制 0-9 模板")
    r.add_argument("--count", type=int, default=3, help="每个数字录制次数")
    r.add_argument("--sr", type=int, default=16000, help="采样率")
    r.add_argument("--duration", type=float, default=1.0, help="每条录音时长（秒）")
    r.add_argument("--data-root", type=str, default="data", help="数据根目录")
    r.set_defaults(func=cmd_record_templates)

    # train
    t = sub.add_parser("train", help="训练最近质心模型")
    t.add_argument("--sr", type=int, default=16000)
    t.add_argument("--n_mfcc", type=int, default=13)
    t.add_argument("--n_mels", type=int, default=26)
    t.add_argument("--frame_ms", type=float, default=25.0)
    t.add_argument("--hop_ms", type=float, default=10.0)
    t.add_argument("--data-root", type=str, default="data")
    t.add_argument("--model-path", type=str, default=MODEL_PATH)
    t.set_defaults(func=cmd_train)

    # recognize-3
    g = sub.add_parser("recognize-3", help="连续三位数字识别")
    g.add_argument("--sr", type=int, default=16000)
    g.add_argument("--n_mfcc", type=int, default=13)
    g.add_argument("--n_mels", type=int, default=26)
    g.add_argument("--frame_ms", type=float, default=25.0)
    g.add_argument("--hop_ms", type=float, default=10.0)
    g.add_argument("--max-seconds", type=float, default=5.0, help="最长录音时长")
    g.add_argument("--model-path", type=str, default=MODEL_PATH)
    g.set_defaults(func=cmd_recognize_3)

    # record-once (utility)
    o = sub.add_parser("record-once", help="快速录音测试")
    o.add_argument("--sr", type=int, default=16000)
    o.add_argument("--duration", type=float, default=1.0)
    o.set_defaults(func=cmd_record_once)

    return p


def main(argv: List[str] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

