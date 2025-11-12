import argparse
from datetime import datetime
from pathlib import Path

from audio_io import countdown, record, save_wav


def main():
    parser = argparse.ArgumentParser(description="录入 0-9 语音数据")
    parser.add_argument("--samples-per-digit", type=int, default=5, help="每个数字录制的样本条数")
    parser.add_argument("--duration", type=float, default=1.2, help="每条样本录音时长(秒)")
    parser.add_argument("--sr", type=int, default=16000, help="采样率")
    parser.add_argument("--device", type=int, default=None, help="输入设备编号（可选）")
    parser.add_argument("--out", type=str, default="data/raw", help="输出根目录")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("将依次录制 0-9，每个数字多条样本。准备好后按提示倒计时开始说话。")
    for digit in range(10):
        print(f"\n=== 录制数字: {digit} ===")
        digit_dir = out_root / str(digit)
        digit_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, args.samples_per_digit + 1):
            print(f"样本 {i}/{args.samples_per_digit}：请在倒计时后清晰读出 ‘{digit}’")
            countdown(3)
            y = record(duration=args.duration, sr=args.sr, device=args.device)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = digit_dir / f"{ts}.wav"
            save_wav(save_path, y, args.sr)
            print(f"已保存: {save_path}")

    print("\n录制完成。可以运行训练脚本开始训练模型。")


if __name__ == "__main__":
    main()

