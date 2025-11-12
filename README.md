# 音频数字识别（0-9，自录模板 + 连读三位数识别）

本项目提供两个模式：
- 输入模式（模板录制）：用户依次录制数字 0-9 的语音样本，作为个性化模板。
- 识别模式（连续三位）：用户连续说三个数字（中间留短暂停顿），程序自动分段并识别三位数字。

不依赖大型框架，使用 Numpy 自实现 MFCC 与简单最近质心分类器；录音依赖 `sounddevice`。

## 环境准备

- Python 3.8+
- 依赖安装（离线网络也可先准备好轮子包）：

```
pip install -r requirements.txt
```

requirements.txt 内容很精简：
- numpy
- sounddevice

如果无法安装/没有麦克风，本项目也支持从 `wav` 文件离线处理（见高级用法）。

## 目录结构

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── app.py                 # 命令行入口
│   ├── audio_utils.py         # 录音/读写 wav
│   ├── features.py            # MFCC 提取（纯 numpy 实现）
│   ├── segment.py             # 基于短时能量+静音检测的分段
│   ├── dataset.py             # 样本集管理、特征提取
│   └── model.py               # 简单最近质心分类器
├── data/
│   └── raw/                   # 录制原始音频：data/raw/<digit>/sample_*.wav
└── models/
    └── centroids.npz          # 训练后保存的类别质心
```

## 快速上手

1) 录制模板（每个数字默认录 3 次，可自定义）：
```
python -m src.app record-templates --count 3 --sr 16000 --duration 1.0
```
按提示依次录 0-9。建议安静环境，清晰朗读，每次之间自然停顿。

2) 训练模型（根据已录制样本计算每类质心）：
```
python -m src.app train
```
成功后在 `models/centroids.npz` 生成模型文件。

3) 连续三位数字识别：
```
python -m src.app recognize-3 --sr 16000 --max-seconds 5
```
提示音后请在 5 秒内说三位数字（中间留短暂停顿，如“3 … 1 … 4”）。程序会自动分段并输出识别结果，例如：
```
识别结果：3 1 4
```

## 一键运行（自动创建虚拟环境）

无需手动安装依赖，使用 bash 启动脚本：

```
bash bootstrap.sh menu
```

或直接走全流程（录制→训练→识别）：

```
bash bootstrap.sh all
```

说明：脚本会在项目根目录创建 `.venv` 虚拟环境，并安装 `requirements.txt` 后调用 `src.app`。

## 常见问题

- 无法导入 sounddevice：
  - 请安装依赖，或检查系统麦克风权限；Linux 需要 `alsa/pulseaudio`，Windows 需启用录音设备。
- 识别率不理想：
  - 录制模板时尽量统一音量、语速；
  - 每个数字多录几条（比如 5-10 条），再 `train`；
  - 识别时每个数字之间留出短暂停顿（>200ms）。
- 想用已有 wav 文件：
  - 将 `wav` 放到 `data/raw/<digit>/` 目录后，直接运行 `train`；文件采样率建议 16k 单声道。

## 高级用法

- 自定义 MFCC 维度与帧设置：
```
python -m src.app train --n_mfcc 13 --frame_ms 25 --hop_ms 10
```
- 仅录音不保存、快速测试：
```
python -m src.app record-once --sr 16000 --duration 1.0
```

## 免责声明

该项目为教学/演示用途，算法简洁，不能保证在嘈杂环境、高速连读情况下的高准确率。
