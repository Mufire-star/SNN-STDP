#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${RAW_DIR:-$ROOT_DIR/data/mnist/raw}"
RUN_DOWNLOAD=1
RUN_HLS=1
RUN_VIVADO=1
VIVADO_JOBS="${VIVADO_JOBS:-1}"
HLS_SKIP_CSIM="${HLS_SKIP_CSIM:-1}"
HLS_CLOCK_NS="${HLS_CLOCK_NS:-10.0}"

usage() {
  cat <<'EOF'
Usage: ./build_stdp.sh [options]

Build the STDP-only project:
  1) Download MNIST raw files
  2) Run Vitis HLS csim/csynth/export
  3) Run Vivado implementation and write bitstream

Options:
  -h, --help          Show this help message
      --raw-dir DIR   MNIST raw directory (default: data/mnist/raw)
      --skip-download Skip dataset download
      --skip-hls      Skip Vitis HLS
      --skip-vivado   Skip Vivado bitstream generation
      --jobs N        Parallel jobs for Vivado (default: 1)
      --clock-ns N    HLS target clock in ns (default: 10.0)
      --run-csim      Run HLS C simulation before synthesis
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --raw-dir)
      RAW_DIR="$2"
      shift 2
      ;;
    --skip-download)
      RUN_DOWNLOAD=0
      shift
      ;;
    --skip-hls)
      RUN_HLS=0
      shift
      ;;
    --skip-vivado)
      RUN_VIVADO=0
      shift
      ;;
    --jobs)
      VIVADO_JOBS="$2"
      shift 2
      ;;
    --clock-ns)
      HLS_CLOCK_NS="$2"
      shift 2
      ;;
    --run-csim)
      HLS_SKIP_CSIM=0
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

log() {
  echo "[$(date '+%F %T')] $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing command: $1" >&2
    exit 1
  }
}

require_cmd python3
if [[ "$RUN_VIVADO" == "1" ]]; then
  require_cmd vivado
fi

if [[ "$RUN_DOWNLOAD" == "1" ]]; then
  log "Downloading MNIST raw dataset"
  python3 "$ROOT_DIR/scripts/download_mnist.py" --out "$RAW_DIR"
fi

for f in \
  "$RAW_DIR/train-images-idx3-ubyte" \
  "$RAW_DIR/train-labels-idx1-ubyte" \
  "$RAW_DIR/t10k-images-idx3-ubyte" \
  "$RAW_DIR/t10k-labels-idx1-ubyte"
do
  [[ -f "$f" ]] || {
    echo "Missing dataset file: $f" >&2
    exit 1
  }
done

export MNIST_RAW_DIR="$RAW_DIR"
export VIVADO_JOBS
export HLS_SKIP_CSIM
export HLS_CLOCK_NS

if [[ "$RUN_HLS" == "1" ]]; then
  log "Running Vitis HLS STDP flow"
  pushd "$ROOT_DIR/HW/stdp_snn" >/dev/null
  if command -v vitis-run >/dev/null 2>&1; then
    vitis-run --mode hls --tcl build_hls_stdp.tcl
  elif command -v vitis_hls >/dev/null 2>&1; then
    vitis_hls -f build_hls_stdp.tcl
  elif command -v vivado_hls >/dev/null 2>&1; then
    vivado_hls -f build_hls_stdp.tcl
  else
    echo "Missing HLS tool: need one of vitis-run, vitis_hls, or vivado_hls" >&2
    exit 1
  fi
  popd >/dev/null
fi

if [[ "$RUN_VIVADO" == "1" ]]; then
  log "Running Vivado bitstream flow"
  vivado -mode batch -source "$ROOT_DIR/HW/stdp_snn/vivado/build_bit_stdp.tcl"

  BIT_OUT="$ROOT_DIR/HW/stdp_snn/vivado/output_stdp/snn_stdp.bit"
  HWH_OUT="$ROOT_DIR/HW/stdp_snn/vivado/output_stdp/snn_stdp.hwh"
  [[ -f "$BIT_OUT" ]] || { echo "Missing bitstream: $BIT_OUT" >&2; exit 1; }
  [[ -f "$HWH_OUT" ]] || { echo "Missing hwh: $HWH_OUT" >&2; exit 1; }
  log "BIT: $BIT_OUT"
  log "HWH: $HWH_OUT"
fi

log "STDP project flow complete"
