#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import struct
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np


MODE_INFER = 0
MODE_TRAIN = 1
IMG_H = 28
IMG_W = 28
IMG_BYTES = IMG_H * IMG_W
OUTPUT_CLASSES = 10
OUTPUT_BYTES = OUTPUT_CLASSES * 2
TRAIN_IMAGE_COUNT = 4

MM2S_DMACR = 0x00
MM2S_DMASR = 0x04
MM2S_SA = 0x18
MM2S_LENGTH = 0x28
S2MM_DMACR = 0x30
S2MM_DMASR = 0x34
S2MM_DA = 0x48
S2MM_LENGTH = 0x58

DMA_CR_RS = 0x00000001
DMA_CR_RESET = 0x00000004
DMA_SR_IDLE = 0x00000002
DMA_SR_ERR_MASK = 0x00000770
DMA_SR_IRQ_MASK = 0x00007000
DMA_SR_ERR_BITS = {
    0x00000010: "DMAIntErr",
    0x00000020: "DMASlvErr",
    0x00000040: "DMADecErr",
    0x00000100: "SGIntErr",
    0x00000200: "SGSlvErr",
    0x00000400: "SGDecErr",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_bit_candidates() -> list[Path]:
    root = repo_root()
    return [
        root / "HW/stdp_snn/vivado/output_stdp/snn_stdp.bit",
        Path.cwd() / "snn_stdp.bit",
        Path(__file__).resolve().parent / "snn_stdp.bit",
    ]


def default_raw_dir_candidates() -> list[Path]:
    root = repo_root()
    return [
        root / "data/mnist/raw",
        Path.cwd() / "data/mnist/raw",
        Path("/home/xilinx/jupyter_notebooks/data/mnist/raw"),
    ]


def resolve_existing_path(path: Path | None, candidates: Sequence[Path], description: str) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"{description} not found: {path}")
        return path

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"could not find {description}; tried:\n{tried}")


def read_idx_images(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051 or rows != IMG_H or cols != IMG_W:
            raise ValueError(f"invalid image file: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"invalid label file: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.shape[0] != num:
        raise ValueError(f"label count mismatch in {path}")
    return data


def split_files(raw_dir: Path, split_name: str) -> tuple[Path, Path]:
    if split_name == "train":
        return raw_dir / "train-images-idx3-ubyte", raw_dir / "train-labels-idx1-ubyte"
    if split_name == "test":
        return raw_dir / "t10k-images-idx3-ubyte", raw_dir / "t10k-labels-idx1-ubyte"
    raise ValueError(f"unsupported split: {split_name}")


def load_split(raw_dir: Path, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    image_path, label_path = split_files(raw_dir, split_name)
    return read_idx_images(image_path), read_idx_labels(label_path)


def load_demo_batch(
    raw_dir: Path,
    train_split: str,
    test_split: str,
    train_count: int,
    train_start: int,
    test_index: int,
) -> tuple[list[np.ndarray], list[int], list[int], np.ndarray, int]:
    train_images_all, train_labels_all = load_split(raw_dir, train_split)
    test_images_all, test_labels_all = load_split(raw_dir, test_split)

    if train_count < 0:
        raise ValueError("train_count must be non-negative")
    if train_start < 0 or test_index < 0:
        raise ValueError("indices must be non-negative")
    if train_start + train_count > train_images_all.shape[0]:
        raise ValueError("training slice exceeds dataset size")
    if test_index >= test_images_all.shape[0]:
        raise ValueError("test_index exceeds dataset size")

    train_indices = list(range(train_start, train_start + train_count))
    train_images = [train_images_all[idx].reshape(-1).astype(np.uint8) for idx in train_indices]
    train_labels = [int(train_labels_all[idx]) for idx in train_indices]
    test_image = test_images_all[test_index].reshape(-1).astype(np.uint8)
    test_label = int(test_labels_all[test_index])
    return train_images, train_labels, train_indices, test_image, test_label


def wait_dma_idle(dma, status_reg: int, timeout_s: float) -> None:
    t0 = time.perf_counter()
    while True:
        sr = dma.read(status_reg)
        if sr & DMA_SR_ERR_MASK:
            flags = [name for bit, name in DMA_SR_ERR_BITS.items() if sr & bit]
            names = ",".join(flags) if flags else "unknown"
            raise RuntimeError(f"DMA error, status=0x{sr:08X}, flags={names}")
        if sr & DMA_SR_IRQ_MASK:
            dma.write(status_reg, sr & DMA_SR_IRQ_MASK)
        if sr & DMA_SR_IDLE:
            return
        if (time.perf_counter() - t0) > timeout_s:
            raise TimeoutError(f"DMA timeout, status=0x{sr:08X}")


def init_dma_channels(dma) -> None:
    dma.write(MM2S_DMACR, DMA_CR_RESET)
    dma.write(S2MM_DMACR, DMA_CR_RESET)

    t0 = time.perf_counter()
    while True:
        mm2s_cr = dma.read(MM2S_DMACR)
        s2mm_cr = dma.read(S2MM_DMACR)
        if ((mm2s_cr & DMA_CR_RESET) == 0) and ((s2mm_cr & DMA_CR_RESET) == 0):
            break
        if (time.perf_counter() - t0) > 0.05:
            raise TimeoutError(
                f"DMA reset timeout, MM2S_CR=0x{mm2s_cr:08X}, S2MM_CR=0x{s2mm_cr:08X}"
            )

    dma.write(MM2S_DMASR, dma.read(MM2S_DMASR) & DMA_SR_IRQ_MASK)
    dma.write(S2MM_DMASR, dma.read(S2MM_DMASR) & DMA_SR_IRQ_MASK)
    dma.write(MM2S_DMACR, DMA_CR_RS)
    dma.write(S2MM_DMACR, DMA_CR_RS)


def build_payload(mode: int, test_image: np.ndarray, train_images: Sequence[np.ndarray]) -> np.ndarray:
    total = 1 + test_image.size
    if mode == MODE_TRAIN:
        total += sum(img.size for img in train_images)

    payload = np.empty((total,), dtype=np.uint8)
    payload[0] = mode
    cursor = 1

    if mode == MODE_TRAIN:
        for img in train_images:
            payload[cursor:cursor + img.size] = img
            cursor += img.size

    payload[cursor:cursor + test_image.size] = test_image
    return payload


def run_once(dma, allocate_fn, payload: np.ndarray, timeout_s: float) -> np.ndarray:
    in_buf = allocate_fn(shape=(payload.size,), dtype=np.uint8)
    out_buf = allocate_fn(shape=(OUTPUT_CLASSES,), dtype=np.uint16)
    try:
        in_buf[:] = payload
        in_buf.flush()
        out_buf[:] = 0
        out_buf.flush()

        dma.write(MM2S_DMACR, DMA_CR_RS)
        dma.write(S2MM_DMACR, DMA_CR_RS)
        dma.write(MM2S_DMASR, dma.read(MM2S_DMASR) & DMA_SR_IRQ_MASK)
        dma.write(S2MM_DMASR, dma.read(S2MM_DMASR) & DMA_SR_IRQ_MASK)

        dma.write(S2MM_DA, out_buf.device_address)
        dma.write(S2MM_LENGTH, OUTPUT_BYTES)
        dma.write(MM2S_SA, in_buf.device_address)
        dma.write(MM2S_LENGTH, int(payload.size))

        wait_dma_idle(dma, MM2S_DMASR, timeout_s)
        wait_dma_idle(dma, S2MM_DMASR, timeout_s)

        out_buf.invalidate()
        return np.array(out_buf, copy=True)
    finally:
        in_buf.freebuffer()
        out_buf.freebuffer()


def topk_pairs(scores: np.ndarray, topk: int = 3) -> list[tuple[int, int]]:
    order = np.argsort(scores)[::-1][:topk]
    return [(int(idx), int(scores[idx])) for idx in order]


def run_mode(
    dma,
    allocate_fn,
    mode: int,
    test_image: np.ndarray,
    train_images: Sequence[np.ndarray],
    repeat: int,
    timeout_s: float,
) -> tuple[np.ndarray, float, bool]:
    payload = build_payload(mode, test_image, train_images)
    scores_runs: list[np.ndarray] = []
    latencies_ms: list[float] = []

    for _ in range(repeat):
        t0 = time.perf_counter()
        scores = run_once(dma, allocate_fn, payload, timeout_s)
        latencies_ms.append((time.perf_counter() - t0) * 1e3)
        scores_runs.append(scores)

    baseline = scores_runs[0]
    consistent = all(np.array_equal(baseline, scores) for scores in scores_runs[1:])
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
    return baseline, avg_latency_ms, consistent


def resolve_dma(overlay, dma_name: str):
    if dma_name in overlay.ip_dict:
        return getattr(overlay, dma_name)

    available = ", ".join(sorted(overlay.ip_dict.keys()))
    raise KeyError(f"{dma_name} not found in overlay; available IPs: {available}")


def print_result_block(
    title: str,
    scores: np.ndarray,
    latency_ms: float,
    repeat: int,
    consistent: bool,
) -> None:
    pred = int(np.argmax(scores))
    print(f"{title}:")
    print(f"  scores = {scores.astype(np.uint16).tolist()}")
    print(f"  top3   = {topk_pairs(scores)}")
    print(f"  pred   = {pred}")
    print(f"  avg_ms = {latency_ms:.3f} (repeat={repeat})")
    print(f"  stable = {'yes' if consistent else 'no'}")


def save_results(
    save_dir: Path,
    tag: str,
    metadata: dict,
    results: dict[str, dict],
) -> tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / f"{tag}_summary.json"
    csv_path = save_dir / f"{tag}_scores.csv"

    summary = {
        "metadata": metadata,
        "results": results,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "class_idx", "score", "predicted", "avg_latency_ms", "stable"])
        for mode_name, result in results.items():
            pred = int(result["predicted_class"])
            avg_latency_ms = float(result["avg_latency_ms"])
            stable = bool(result["stable"])
            for class_idx, score in enumerate(result["scores"]):
                writer.writerow([mode_name, class_idx, score, pred, f"{avg_latency_ms:.3f}", int(stable)])

    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the STDP overlay on a PYNQ board")
    parser.add_argument("--bit", type=Path, default=None, help="Path to .bit file")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory with MNIST raw files")
    parser.add_argument("--mode", choices=["infer", "train", "both"], default="both")
    parser.add_argument("--train-split", choices=["train", "test"], default="train")
    parser.add_argument("--test-split", choices=["train", "test"], default="test")
    parser.add_argument("--train-count", type=int, default=TRAIN_IMAGE_COUNT)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--dma-ip", default="axi_dma_0")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory to save json/csv results")
    parser.add_argument("--tag", default=None, help="Prefix for saved result files")
    args = parser.parse_args()

    if args.repeat <= 0:
        raise ValueError("repeat must be positive")

    if args.mode in {"train", "both"} and args.train_count != TRAIN_IMAGE_COUNT:
        raise ValueError(
            f"train_count must be exactly {TRAIN_IMAGE_COUNT} for the current hardware build"
        )

    bit_path = resolve_existing_path(args.bit, default_bit_candidates(), "bitstream")
    hwh_path = bit_path.with_suffix(".hwh")
    if not hwh_path.exists():
        raise FileNotFoundError(f"hwh not found next to bitstream: {hwh_path}")

    raw_dir = resolve_existing_path(args.raw_dir, default_raw_dir_candidates(), "MNIST raw directory")
    train_images, train_labels, train_indices, test_image, test_label = load_demo_batch(
        raw_dir=raw_dir,
        train_split=args.train_split,
        test_split=args.test_split,
        train_count=args.train_count if args.mode in {"train", "both"} else 0,
        train_start=args.train_start,
        test_index=args.test_index,
    )

    from pynq import Overlay, allocate

    overlay = Overlay(str(bit_path))
    dma = resolve_dma(overlay, args.dma_ip)
    init_dma_channels(dma)

    print("\n========== STDP PYNQ Validation ==========")
    print(f"Overlay bit : {bit_path}")
    print(f"Overlay hwh : {hwh_path}")
    print(f"DMA IP      : {args.dma_ip}")
    print(f"MNIST raw   : {raw_dir}")
    print(f"Test split  : {args.test_split}")
    print(f"Test index  : {args.test_index}")
    print(f"Test label  : {test_label}")
    if args.mode in {"train", "both"}:
        print(f"Train split : {args.train_split}")
        print(f"Train idx   : {train_indices}")
        print(f"Train label : {train_labels}")
    print("==========================================")

    exit_code = 0
    results: dict[str, dict] = {}

    if args.mode in {"infer", "both"}:
        infer_scores, infer_ms, infer_consistent = run_mode(
            dma=dma,
            allocate_fn=allocate,
            mode=MODE_INFER,
            test_image=test_image,
            train_images=[],
            repeat=args.repeat,
            timeout_s=args.timeout_s,
        )
        print_result_block("Infer-only", infer_scores, infer_ms, args.repeat, infer_consistent)
        results["infer"] = {
            "scores": infer_scores.astype(np.uint16).tolist(),
            "top3": topk_pairs(infer_scores),
            "predicted_class": int(np.argmax(infer_scores)),
            "avg_latency_ms": infer_ms,
            "stable": infer_consistent,
        }
        if not infer_consistent:
            exit_code = 2

    if args.mode in {"train", "both"}:
        train_scores, train_ms, train_consistent = run_mode(
            dma=dma,
            allocate_fn=allocate,
            mode=MODE_TRAIN,
            test_image=test_image,
            train_images=train_images,
            repeat=args.repeat,
            timeout_s=args.timeout_s,
        )
        print_result_block("Train-then-infer", train_scores, train_ms, args.repeat, train_consistent)
        results["train_then_infer"] = {
            "scores": train_scores.astype(np.uint16).tolist(),
            "top3": topk_pairs(train_scores),
            "predicted_class": int(np.argmax(train_scores)),
            "avg_latency_ms": train_ms,
            "stable": train_consistent,
        }
        if not train_consistent:
            exit_code = 2

    if args.save_dir is not None:
        tag = args.tag or datetime.now().strftime("stdp_run_%Y%m%d_%H%M%S")
        metadata = {
            "overlay_bit": str(bit_path),
            "overlay_hwh": str(hwh_path),
            "raw_dir": str(raw_dir),
            "dma_ip": args.dma_ip,
            "mode": args.mode,
            "train_split": args.train_split,
            "test_split": args.test_split,
            "train_count": args.train_count if args.mode in {"train", "both"} else 0,
            "train_start": args.train_start,
            "train_indices": train_indices if args.mode in {"train", "both"} else [],
            "train_labels": train_labels if args.mode in {"train", "both"} else [],
            "test_index": args.test_index,
            "test_label": test_label,
            "repeat": args.repeat,
            "timeout_s": args.timeout_s,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        json_path, csv_path = save_results(args.save_dir, tag, metadata, results)
        print(f"Saved summary: {json_path}")
        print(f"Saved scores : {csv_path}")

    print("==========================================\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
