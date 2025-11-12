#!/usr/bin/env bash
set -euo pipefail

# One-click bootstrap for this project (bash).
# - Creates/uses a local virtualenv (default: .venv)
# - Installs requirements
# - Runs the app in selected mode
#
# Usage examples:
#   bash bootstrap.sh menu
#   bash bootstrap.sh recognize-3 -- --utterance-manual --decode dtw
#   bash bootstrap.sh --venv .venv2 train

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR=".venv"
MODE="menu"
SKIP_INSTALL=0
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: bash bootstrap.sh [OPTIONS] [menu|record-templates|record-digit|train|recognize-1|recognize-3] [-- ARGS]

Options:
  --venv DIR         Virtualenv directory (default: .venv)
  --skip-install     Skip pip install step

Examples:
  bash bootstrap.sh menu
  bash bootstrap.sh recognize-3 -- --utterance-manual --decode dtw
  bash bootstrap.sh --venv .venv2 train
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      [[ $# -ge 2 ]] || { echo "--venv requires a value" >&2; exit 2; }
      VENV_DIR="$2"; shift 2 ;;
    --skip-install)
      SKIP_INSTALL=1; shift ;;
    menu|record-templates|record-digit|train|recognize-1|recognize-3)
      MODE="$1"; shift ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }

VENV_PY="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "[bootstrap] Creating venv at $VENV_DIR …"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[bootstrap] Using existing venv: $VENV_DIR"
fi

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  echo "[bootstrap] Upgrading pip …"
  "$VENV_PY" -m pip install --upgrade pip wheel || echo "[bootstrap] pip upgrade failed (continuing)"
  if [[ -f requirements.txt ]]; then
    echo "[bootstrap] Installing requirements …"
    "$VENV_PY" -m pip install -r requirements.txt
  else
    echo "[bootstrap] requirements.txt not found; skipping installs."
  fi
else
  echo "[bootstrap] Skip install per flag."
fi

run_app() {
  "$VENV_PY" -m src.app "$@"
}

case "$MODE" in
  menu)
    while true; do
      echo
      echo "== 语音数字识别 =="
      echo "1) 录制模板 0-9（手动开始/结束）"
      echo "2) 录制某一数字（手动，增量样本）"
      echo "3) 训练模型"
      echo "4) 一次性读三位（手动，DTW 模式）"
      echo "5) 单位识别（手动）"
      echo "q) 退出"
      read -rp "选择: " choice
      case "$choice" in
        1) run_app record-templates --count 3 --sr 16000 ;;
        2)
          read -rp "输入要录制的数字(0-9): " digit
          if [[ ! "$digit" =~ ^[0-9]$ ]]; then echo "无效数字"; continue; fi
          read -rp "录制次数(默认3): " cnt; cnt=${cnt:-3}
          while true; do
            run_app record-digit --digit "$digit" --count "$cnt" --sr 16000
            read -rp "是否继续为该数字追加样本？[y/N]: " yn
            case "$yn" in
              y|Y) ;;
              *) break ;;
            esac
          done
          read -rp "是否现在训练模型？[y/N]: " yn
          case "$yn" in
            y|Y) run_app train ;;
            *) : ;;
          esac
          ;;
        3) run_app train ;;
        4)
          while true; do
            echo "按回车开始，一次性读出三个数字，读完再按回车结束…"
            run_app recognize-3 --sr 16000 --utterance-manual --decode dtw
            read -rp "继续识别该模式？[y/N]: " yn
            case "$yn" in
              y|Y) ;;
              *) break ;;
            esac
          done
          ;;
        5)
          while true; do
            echo "按回车开始，说出一个数字，读完再按回车结束…"
            run_app recognize-1 --sr 16000 --manual
            read -rp "继续识别该模式？[y/N]: " yn
            case "$yn" in
              y|Y) ;;
              *) break ;;
            esac
          done
          ;;
        q|Q) exit 0 ;;
        *) echo "无效选择，请重试。" ;;
      esac
    done
    ;;
  record-templates)
    run_app record-templates "${EXTRA_ARGS[@]}" ;;
  train)
    run_app train "${EXTRA_ARGS[@]}" ;;
  recognize-3)
    run_app recognize-3 "${EXTRA_ARGS[@]}" ;;
  recognize-1)
    run_app recognize-1 "${EXTRA_ARGS[@]}" ;;
  all)
    run_app record-templates --count 3 --sr 16000 && \
    run_app train && \
    run_app recognize-3 --sr 16000 --max-seconds 5 ;;
esac
