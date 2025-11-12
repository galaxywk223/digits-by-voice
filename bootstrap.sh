#!/usr/bin/env bash
set -euo pipefail

# One-click bootstrap for this project (bash).
# - Creates/uses a local virtualenv (default: .venv)
# - Installs requirements
# - Runs the app in selected mode
#
# Usage examples:
#   bash bootstrap.sh menu
#   bash bootstrap.sh all
#   bash bootstrap.sh recognize-3 -- --max-seconds 5
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
Usage: bash bootstrap.sh [OPTIONS] [menu|record-templates|train|recognize-3|all] [-- ARGS]

Options:
  --venv DIR         Virtualenv directory (default: .venv)
  --skip-install     Skip pip install step

Examples:
  bash bootstrap.sh menu
  bash bootstrap.sh all
  bash bootstrap.sh recognize-3 -- --max-seconds 5
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
    menu|record-templates|train|recognize-3|all)
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
      echo "== Voice Digits =="
      echo "1) 录制模板 (0-9)"
      echo "2) 训练模型"
      echo "3) 连续三位识别"
      echo "4) 一键 (1 -> 2 -> 3)"
      echo "q) 退出"
      read -rp "选择: " choice
      case "$choice" in
        1) run_app record-templates --count 3 --sr 16000 --duration 1.0 ;;
        2) run_app train ;;
        3) run_app recognize-3 --sr 16000 --max-seconds 5 ;;
        4)
          run_app record-templates --count 3 --sr 16000 --duration 1.0 && \
          run_app train && \
          run_app recognize-3 --sr 16000 --max-seconds 5 ;;
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
  all)
    run_app record-templates --count 3 --sr 16000 --duration 1.0 && \
    run_app train && \
    run_app recognize-3 --sr 16000 --max-seconds 5 ;;
esac

