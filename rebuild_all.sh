#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-FPGA_ML}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_QUANT="${RUN_QUANT:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
TRAIN_DEVICE="${TRAIN_DEVICE:-}"
QUANT_METHOD="${QUANT_METHOD:-static}"
QUANT_DEVICE="${QUANT_DEVICE:-cpu}"

usage() {
  cat <<'EOF'
Usage: ./rebuild_all.sh [options]

Rebuild full flow: dataset -> train -> quantize -> export weights -> HLS -> bitstream.

Options:
  -h, --help                 Show this help message and exit
      --env-name NAME        Conda environment name (default: FPGA_ML)

      --run-train            Enable training step (default)
      --no-train             Disable training step
      --train-epochs N       Training epochs (default: 10)
      --train-device DEV     Training device, e.g. cpu/cuda (default: auto in train.py)

      --run-quant            Enable quantization step (default)
      --no-quant             Disable quantization step
      --quant-method M       Quant method: static|dynamic|qat (default: static)
      --quant-device DEV     Quantization device (default: cpu)

Examples:
  ./rebuild_all.sh
  ./rebuild_all.sh --no-train --no-quant
  ./rebuild_all.sh --train-epochs 20 --train-device cpu --quant-method static
  ./rebuild_all.sh --env-name FPGA_ML --run-train --run-quant
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --env-name)
      [[ $# -ge 2 ]] || { echo "ERROR: --env-name requires a value" >&2; exit 2; }
      ENV_NAME="$2"
      shift 2
      ;;
    --run-train)
      RUN_TRAIN=1
      shift
      ;;
    --no-train)
      RUN_TRAIN=0
      shift
      ;;
    --train-epochs)
      [[ $# -ge 2 ]] || { echo "ERROR: --train-epochs requires a value" >&2; exit 2; }
      TRAIN_EPOCHS="$2"
      shift 2
      ;;
    --train-device)
      [[ $# -ge 2 ]] || { echo "ERROR: --train-device requires a value" >&2; exit 2; }
      TRAIN_DEVICE="$2"
      shift 2
      ;;
    --run-quant)
      RUN_QUANT=1
      shift
      ;;
    --no-quant)
      RUN_QUANT=0
      shift
      ;;
    --quant-method)
      [[ $# -ge 2 ]] || { echo "ERROR: --quant-method requires a value" >&2; exit 2; }
      QUANT_METHOD="$2"
      shift 2
      ;;
    --quant-device)
      [[ $# -ge 2 ]] || { echo "ERROR: --quant-device requires a value" >&2; exit 2; }
      QUANT_DEVICE="$2"
      shift 2
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$RUN_TRAIN" =~ ^[01]$ ]]; then
  echo "ERROR: RUN_TRAIN must be 0 or 1, got '$RUN_TRAIN'" >&2
  exit 2
fi
if ! [[ "$RUN_QUANT" =~ ^[01]$ ]]; then
  echo "ERROR: RUN_QUANT must be 0 or 1, got '$RUN_QUANT'" >&2
  exit 2
fi
if ! [[ "$TRAIN_EPOCHS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: TRAIN_EPOCHS must be a non-negative integer, got '$TRAIN_EPOCHS'" >&2
  exit 2
fi
if [[ "$QUANT_METHOD" != "static" && "$QUANT_METHOD" != "dynamic" && "$QUANT_METHOD" != "qat" ]]; then
  echo "ERROR: QUANT_METHOD must be one of: static|dynamic|qat, got '$QUANT_METHOD'" >&2
  exit 2
fi

HLS_DIR="$ROOT_DIR/HW/baseline/snn_ip"
CKPT="$ROOT_DIR/python/SNN-baseline/weights/fp32/mnist_snn_baseline.pt"
Q_SAVE_DIR="$ROOT_DIR/python/SNN-baseline/weights/int8"
WEIGHTS_HDR="$HLS_DIR/weights/weights_generated.h"
HLS_TCL="$HLS_DIR/build_snn_ip_v2_conv2.tcl"
VIVADO_TCL="$HLS_DIR/vivado/build_pynq_bit_v2.tcl"
BIT_OUT="$HLS_DIR/vivado/output_v2/snn_v2.bit"
HWH_OUT="$HLS_DIR/vivado/output_v2/snn_v2.hwh"
MNIST_RAW_DIR="$ROOT_DIR/python/SNN-baseline/datas/MNIST/raw"

log() {
  echo "[$(date '+%F %T')] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: command not found: $1" >&2
    exit 1
  fi
}

log "Checking required tools"
require_cmd conda
require_cmd vitis-run
require_cmd vivado

if [[ ! -f "$HLS_TCL" ]]; then
  echo "ERROR: HLS TCL not found: $HLS_TCL" >&2
  exit 1
fi
if [[ ! -f "$VIVADO_TCL" ]]; then
  echo "ERROR: Vivado TCL not found: $VIVADO_TCL" >&2
  exit 1
fi

ensure_mnist() {
  local req=(
    "train-images-idx3-ubyte"
    "train-labels-idx1-ubyte"
    "t10k-images-idx3-ubyte"
    "t10k-labels-idx1-ubyte"
  )

  local missing=0
  for f in "${req[@]}"; do
    if [[ ! -f "$MNIST_RAW_DIR/$f" ]]; then
      missing=1
      break
    fi
  done

  if [[ "$missing" -eq 0 ]]; then
    log "MNIST raw files found"
    return 0
  fi

  log "MNIST raw files missing, downloading dataset"
  conda run -n "$ENV_NAME" python - <<PY
from pathlib import Path
from torchvision import datasets

root = Path(r"$ROOT_DIR/python/SNN-baseline/datas")
root.mkdir(parents=True, exist_ok=True)
datasets.MNIST(root=str(root), train=True, download=True)
datasets.MNIST(root=str(root), train=False, download=True)
print("MNIST ready at", root / "MNIST" / "raw")
PY
}

log "Step 1/5: Ensuring MNIST dataset"
ensure_mnist

if [[ "$RUN_TRAIN" == "1" ]]; then
  log "Step 2/5: Training baseline model"
  train_cmd=(
    conda run -n "$ENV_NAME" python "$ROOT_DIR/python/SNN-baseline/train.py"
    --data-dir "$MNIST_RAW_DIR"
    --epochs "$TRAIN_EPOCHS"
    --save "$CKPT"
  )
  if [[ -n "$TRAIN_DEVICE" ]]; then
    train_cmd+=(--device "$TRAIN_DEVICE")
  fi
  "${train_cmd[@]}"
else
  log "Step 2/5: Skip training (RUN_TRAIN=$RUN_TRAIN)"
fi

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found after training step: $CKPT" >&2
  exit 1
fi

if [[ "$RUN_QUANT" == "1" ]]; then
  log "Step 3/5: Quantizing model ($QUANT_METHOD)"
  conda run -n "$ENV_NAME" python "$ROOT_DIR/python/SNN-baseline/quantize.py" \
    --data-dir "$MNIST_RAW_DIR" \
    --ckpt "$CKPT" \
    --method "$QUANT_METHOD" \
    --save-dir "$Q_SAVE_DIR" \
    --device "$QUANT_DEVICE"
else
  log "Step 3/5: Skip quantization (RUN_QUANT=$RUN_QUANT)"
fi

log "Step 4/5: Exporting SNN weights header"
conda run -n "$ENV_NAME" python "$ROOT_DIR/HW/baseline/export_snn_weights.py" \
  --ckpt "$CKPT" \
  --out "$WEIGHTS_HDR"

log "Step 5/5: Running Vitis HLS (v2 conv2 solution)"
pushd "$HLS_DIR" >/dev/null
vitis-run --mode hls --tcl "$HLS_TCL"
popd >/dev/null

log "Final: Running Vivado to generate bitstream/hwh"
vivado -mode batch -source "$VIVADO_TCL"

if [[ ! -f "$BIT_OUT" || ! -f "$HWH_OUT" ]]; then
  echo "ERROR: build finished but output files are missing:" >&2
  echo "  $BIT_OUT" >&2
  echo "  $HWH_OUT" >&2
  exit 1
fi

log "Done"
log "BIT: $BIT_OUT"
log "HWH: $HWH_OUT"
