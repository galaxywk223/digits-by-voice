import argparse
import json
from pathlib import Path

import numpy as np
import joblib
import librosa
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from features import extract_features


def load_dataset(data_dir: Path, sr: int):
    X = []
    y = []
    total = 0
    for digit in range(10):
        digit_dir = data_dir / str(digit)
        if not digit_dir.exists():
            continue
        for wav in sorted(digit_dir.glob("*.wav")):
            try:
                audio, _ = librosa.load(wav, sr=sr, mono=True)
                feat = extract_features(audio, sr)
                X.append(feat)
                y.append(digit)
                total += 1
            except Exception as e:
                print(f"跳过损坏文件: {wav} ({e})")
    if total == 0:
        raise RuntimeError(f"未在 {data_dir} 下找到任何 wav 文件。请先运行 record_digits.py 录制样本。")
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="训练 0-9 数字分类模型")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="训练数据根目录")
    parser.add_argument("--model-path", type=str, default="models/digit_clf.joblib", help="模型保存路径")
    parser.add_argument("--sr", type=int, default=16000, help="采样率")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"加载数据集: {data_dir}")
    X, y = load_dataset(data_dir, sr=args.sr)
    print(f"样本数: {len(y)}，特征维度: {X.shape[1]}")

    # 简单的标准化 + 逻辑回归
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, n_jobs=None))
    ])

    print("开始训练...")
    clf.fit(X, y)
    print("训练完成。")

    joblib.dump(clf, model_path)
    meta = {
        "sr": args.sr,
        "classes": list(range(10)),
        "features": {
            "type": "mfcc+delta+delta2_stats",
            "dim": int(X.shape[1])
        }
    }
    with open(model_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"已保存模型到: {model_path}")


if __name__ == "__main__":
    main()

