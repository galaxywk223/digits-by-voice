import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np

from .audio_utils import (
    beep,
    countdown,
    play_audio,
    read_wav,
    record_audio,
    record_audio_manual,
    write_wav,
)
from .dataset import load_features
from .features import mfcc_mean_feature
from .model import MODEL_PATH, load_model, predict_digit, save_model, train_centroid_model
from .decode import framewise_decode_three, dp_decode_three
from .dtw_decode import dtw_decode_three
from .segment import split_on_silence, force_split_into_n, extract_main_speech, segment_into_three


def cmd_record_templates(args: argparse.Namespace) -> None:
    sr = args.sr
    count = args.count
    dur = args.duration
    out_root = os.path.join(args.data_root, "raw")
    digits = [str(i) for i in range(10)]

    mode = "自动定长" if args.auto else "手动开始/结束"
    print(f"采样率: {sr} Hz, 模式: {mode}，每个数字录制: {count} 条")
    for d in digits:
        ddir = os.path.join(out_root, d)
        os.makedirs(ddir, exist_ok=True)
        print(f"\n请读数字：{d}")
        for i in range(count):
            if args.auto:
                input("按回车开始倒计时…")
                play_audio(beep(sr))
                countdown(0.8)
                audio = record_audio(dur, sr=sr)
            else:
                print("按回车开始录音，录完后再按回车结束…")
                audio = record_audio_manual(sr=sr)
            ts = int(time.time() * 1000)
            path = os.path.join(ddir, f"sample_{ts}_{i+1}.wav")
            write_wav(path, audio, sr)
            print(f"已保存: {path}")
    print("\n模板录制完成！")


def cmd_record_digit(args: argparse.Namespace) -> None:
    sr = args.sr
    count = args.count
    dur = args.duration
    d = int(args.digit)
    if d < 0 or d > 9:
        print("digit 必须是 0-9 之间的整数")
        sys.exit(2)
    out_root = os.path.join(args.data_root, "raw")
    ddir = os.path.join(out_root, str(d))
    os.makedirs(ddir, exist_ok=True)

    mode = "自动定长" if args.auto else "手动开始/结束"
    print(f"录制数字 {d}：采样率 {sr} Hz，模式 {mode}，次数 {count}")
    for i in range(count):
        if args.auto:
            input("按回车开始倒计时…")
            play_audio(beep(sr))
            countdown(0.8)
            audio = record_audio(dur, sr=sr)
        else:
            print("按回车开始录音，录完后再按回车结束…")
            audio = record_audio_manual(sr=sr)
        ts = int(time.time() * 1000)
        path = os.path.join(ddir, f"sample_{ts}_{i+1}.wav")
        write_wav(path, audio, sr)
        print(f"已保存: {path}")
    print("完成！可运行 train 重新训练模型。")


def cmd_train(args: argparse.Namespace) -> None:
    # 兼容传入 data 或 data/raw
    base = os.path.normpath(args.data_root)
    raw_root = base if os.path.basename(base) == "raw" else os.path.join(base, "raw")
    print(f"样本目录：{raw_root}")
    X, y = load_features(
        sr=args.sr,
        n_mfcc=args.n_mfcc,
        n_mels=args.n_mels,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        data_root=raw_root,
    )
    # Framewise features for more robust decoding
    from .dataset import load_framewise_features
    XF, yF = load_framewise_features(
        sr=args.sr,
        n_mfcc=args.n_mfcc,
        n_mels=args.n_mels,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        data_root=raw_root,
    )
    if X.shape[0] == 0:
        print("未发现样本，请先运行 record-templates 或检查目录是否包含 data/raw/0..9 及 wav 文件。")
        sys.exit(1)
    model = train_centroid_model(X, y, XF, yF)
    params = dict(sr=args.sr, n_mfcc=args.n_mfcc, n_mels=args.n_mels, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    save_model(args.model_path, model, params)
    # 打印每类样本数量，便于排查不均衡/缺失
    try:
        counts = np.bincount(y, minlength=10)
    except Exception:
        counts = None
    print(f"模型已保存：{args.model_path}")
    if counts is not None:
        print("样本计数：", " ".join(f"{i}:{int(c)}" for i, c in enumerate(counts)))
        missing = [str(i) for i, c in enumerate(counts) if c == 0]
        if missing:
            print("注意：以下类别没有样本 →", ", ".join(missing))


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

    if args.manual:
        digits = []
        for idx in range(3):
            print(f"第 {idx+1} 个数字：按回车开始，读完再按回车结束…")
            seg = record_audio_manual(sr=sr)
            # 轻裁剪首尾静音，保持与单次识别一致的处理
            seg = extract_main_speech(seg, sr=sr, frame_ms=frame_ms, hop_ms=hop_ms)
            if seg.size < int(0.12 * sr):
                print("录音过短，请重试该位数字。")
                return
            feat = mfcc_mean_feature(seg, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, frame_ms=frame_ms, hop_ms=hop_ms)
            digits.append(predict_digit(feat, model))
        print("识别结果：", " ".join(str(d) for d in digits))
        return

    if args.utterance_manual:
        print("按回车开始，一次性读出三个数字，读完再按回车结束…")
        audio = record_audio_manual(sr=sr)
    else:
        print(f"请在 {args.max_seconds}s 内连续说三个数字（中间留短暂停顿）…")
        play_audio(beep(sr, dur=0.2))
        audio = record_audio(args.max_seconds, sr=sr)

    if args.decode == "frame":
        digits, segs = framewise_decode_three(
            audio,
            sr=sr,
            model=model,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
        )
    elif args.decode == "dp":
        digits, segs = dp_decode_three(
            audio,
            sr=sr,
            model=model,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
        )
    elif args.decode == "dtw":
        # Use recorded templates from data/raw for template matching
        data_root = os.path.join("data", "raw")
        digits, segs = dtw_decode_three(
            audio,
            sr=sr,
            data_root=data_root,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
        )
    else:
        # energy-based segmentation path
        if args.utterance_manual:
            segs = segment_into_three(
                audio,
                sr=sr,
                frame_ms=frame_ms,
                hop_ms=hop_ms,
                min_speech_ms=args.min_speech_ms,
                min_silence_ms=args.min_silence_ms,
                pad_ms=args.pad_ms,
            )
        else:
            segs = split_on_silence(
                audio,
                sr=sr,
                frame_ms=frame_ms,
                hop_ms=hop_ms,
                min_speech_ms=args.min_speech_ms,
                min_silence_ms=args.min_silence_ms,
                pad_ms=args.pad_ms,
                thr_weight=args.vad_weight,
            )
            segs = [seg for seg in segs if (seg[1] - seg[0]) > int(sr * 0.10)]  # 丢弃过短片段
            segs = pick_three_segments(segs, audio)

    if args.decode == "frame":
        if len(digits) != 3:
            print("未能解析出三位，请重试或调整说话停顿。")
            sys.exit(2)
    elif len(segs) != 3:
        # Try a fallback: force split into 3 segments on the whole utterance
        fallback = force_split_into_n(audio, sr=sr, n=3, frame_ms=frame_ms, hop_ms=hop_ms)
        if len(fallback) == 3:
            segs = fallback
        else:
            print(f"未能检测到三段语音（检测到 {len(segs)} 段），请尝试以下方法：")
            print("- 说每个数字之间留更明显的停顿（>200ms）")
            print("- 增大录音窗口或降低门限：例如 --vad-weight 0.15 --min-speech-ms 100")
            print("- 或使用手动模式：recognize-3 --manual（逐位回车开始/结束）")
            # 打印调试信息
            for i, (s, e) in enumerate(segs):
                print(f"段{i+1}: {s/sr:.2f}s - {e/sr:.2f}s, 长度={(e-s)/sr:.2f}s")
            sys.exit(2)

    if args.decode != "frame":
        digits = []
        for i, (s, e) in enumerate(segs):
            seg = audio[s:e]
            # Light trim to remove residual silence inside the segment
            seg = extract_main_speech(seg, sr=sr, frame_ms=frame_ms, hop_ms=hop_ms)
            feat = mfcc_mean_feature(seg, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, frame_ms=frame_ms, hop_ms=hop_ms)
            pred = predict_digit(feat, model)
            digits.append(pred)
    print("识别结果：", " ".join(str(d) for d in digits))


def cmd_record_once(args: argparse.Namespace) -> None:
    if args.auto:
        print(f"试录 {args.duration}s …")
        play_audio(beep(args.sr))
        countdown(0.8)
        audio = record_audio(args.duration, sr=args.sr)
    else:
        print("手动录音模式：按回车开始，录完再按回车结束。")
        audio = record_audio_manual(sr=args.sr)
    os.makedirs("tmp", exist_ok=True)
    out = os.path.join("tmp", "test.wav")
    write_wav(out, audio, args.sr)
    print(f"已保存：{out}")


def cmd_recognize_1(args: argparse.Namespace) -> None:
    model = load_model(args.model_path)
    if model is None:
        print("未找到模型，请先运行 train 训练。")
        sys.exit(1)

    sr = int(model["params"].get("sr", args.sr))
    n_mfcc = int(model["params"].get("n_mfcc", args.n_mfcc))
    n_mels = int(model["params"].get("n_mels", args.n_mels))
    frame_ms = float(model["params"].get("frame_ms", args.frame_ms))
    hop_ms = float(model["params"].get("hop_ms", args.hop_ms))

    if args.manual:
        print("按回车开始录音，读完该数字后再按回车结束…")
        y = record_audio_manual(sr=sr)
    else:
        print(f"将录制 {args.duration}s，请在提示音后说出一个数字…")
        play_audio(beep(sr, dur=0.2))
        countdown(0.8)
        y = record_audio(args.duration, sr=sr)

    # 轻裁剪首尾静音，避免无声干扰
    y = extract_main_speech(y, sr=sr, frame_ms=frame_ms, hop_ms=hop_ms)

    if y.size < int(0.10 * sr):
        print("录音过短，请重试。")
        sys.exit(2)

    feat = mfcc_mean_feature(y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, frame_ms=frame_ms, hop_ms=hop_ms)
    pred = predict_digit(feat, model)
    print("识别结果：", pred)
    if getattr(args, "show_dists", False):
        cents = model["centroids"]; mask = model["mask"]
        valid = np.where(mask)[0]
        dists = np.linalg.norm(cents[valid] - feat[None, :], axis=1)
        pairs = list(zip(valid.tolist(), dists.tolist()))
        pairs.sort(key=lambda x: x[1])
        print("距离：", ", ".join(f"{i}:{d:.3f}" for i, d in pairs))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="音频数字识别（模板+三位连读）")
    sub = p.add_subparsers(dest="cmd", required=True)

    # record-templates
    r = sub.add_parser("record-templates", help="录制 0-9 模板")
    r.add_argument("--count", type=int, default=3, help="每个数字录制次数")
    r.add_argument("--sr", type=int, default=16000, help="采样率")
    r.add_argument("--duration", type=float, default=1.0, help="仅在 --auto 下有效：每条录音时长（秒）")
    r.add_argument("--auto", action="store_true", help="启用自动定长录制（默认手动开始/结束）")
    r.add_argument("--data-root", type=str, default="data", help="数据根目录")
    r.set_defaults(func=cmd_record_templates)

    # record-digit (incremental for a single class)
    rd = sub.add_parser("record-digit", help="增量录制某个数字")
    rd.add_argument("--digit", type=int, required=True, help="要录制的数字 0-9")
    rd.add_argument("--count", type=int, default=3, help="录制次数")
    rd.add_argument("--sr", type=int, default=16000, help="采样率")
    rd.add_argument("--duration", type=float, default=1.0, help="仅在 --auto 下有效：每条录音时长（秒）")
    rd.add_argument("--auto", action="store_true", help="启用自动定长录制（默认手动开始/结束）")
    rd.add_argument("--data-root", type=str, default="data", help="数据根目录")
    rd.set_defaults(func=cmd_record_digit)

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
    g.add_argument("--frame_ms", type=float, default=25.0, help="特征/分段帧长(ms)")
    g.add_argument("--hop_ms", type=float, default=10.0, help="特征/分段帧移(ms)")
    g.add_argument("--max-seconds", type=float, default=5.0, help="最长录音时长")
    g.add_argument("--vad-weight", type=float, default=0.25, help="分段门限权重，越小越敏感(0..1)，仅自动录制时有效")
    g.add_argument("--min-speech-ms", type=float, default=150.0, help="最短语音段时长(ms)")
    g.add_argument("--min-silence-ms", type=float, default=120.0, help="合并段的最短静音(ms)")
    g.add_argument("--pad-ms", type=float, default=50.0, help="分段两端填充(ms)")
    g.add_argument("--utterance-manual", action="store_true", help="一次性读三位：手动开始/结束录音")
    g.add_argument("--decode", choices=["segment", "frame", "dp", "dtw"], default="segment", help="一次性读三位的解码方式：segment=能量分段, frame=逐帧分类, dp=逐帧高斯+动态规划, dtw=模板DTW匹配")
    g.add_argument("--manual", action="store_true", help="手动逐位录音识别（每位按回车开始/结束）")
    g.add_argument("--model-path", type=str, default=MODEL_PATH)
    g.set_defaults(func=cmd_recognize_3)

    # recognize-1
    g1 = sub.add_parser("recognize-1", help="单个数字识别")
    g1.add_argument("--sr", type=int, default=16000)
    g1.add_argument("--n_mfcc", type=int, default=13)
    g1.add_argument("--n_mels", type=int, default=26)
    g1.add_argument("--frame_ms", type=float, default=25.0)
    g1.add_argument("--hop_ms", type=float, default=10.0)
    g1.add_argument("--duration", type=float, default=1.2, help="仅在 --auto 下有效：单次录音时长")
    g1.add_argument("--manual", action="store_true", help="手动开始/结束录音（默认推荐）")
    g1.add_argument("--model-path", type=str, default=MODEL_PATH)
    g1.add_argument("--show-dists", action="store_true", help="显示到各类别质心的距离")
    g1.set_defaults(func=cmd_recognize_1)

    # record-once (utility)
    o = sub.add_parser("record-once", help="快速录音测试")
    o.add_argument("--sr", type=int, default=16000)
    o.add_argument("--duration", type=float, default=1.0, help="仅在 --auto 下有效")
    o.add_argument("--auto", action="store_true", help="启用自动定长录音（默认手动开始/结束）")
    o.set_defaults(func=cmd_record_once)

    return p


def main(argv: List[str] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
