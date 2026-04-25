#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from stdp_overlay_demo import (
    MODE_INFER,
    MODE_TRAIN,
    TRAIN_IMAGE_COUNT,
    default_bit_candidates,
    default_raw_dir_candidates,
    init_dma_channels,
    load_split,
    resolve_existing_path,
    resolve_dma,
    run_mode,
)


def label_indices(labels: np.ndarray) -> dict[int, list[int]]:
    buckets: dict[int, list[int]] = {label: [] for label in range(10)}
    for idx, label in enumerate(labels.tolist()):
        buckets[int(label)].append(idx)
    return buckets


def select_test_cases(labels: np.ndarray, samples_per_class: int, start_offset: int) -> list[int]:
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    if start_offset < 0:
        raise ValueError("start_offset must be non-negative")

    buckets = label_indices(labels)
    selected: list[int] = []
    for label in range(10):
        candidates = buckets[label]
        needed = candidates[start_offset:start_offset + samples_per_class]
        if len(needed) < samples_per_class:
            raise ValueError(
                f"not enough test samples for label {label}: need {samples_per_class}, "
                f"have {max(0, len(candidates) - start_offset)} after offset"
            )
        selected.extend(needed)
    return selected


def choose_support_indices(
    labels: np.ndarray,
    support_label: int,
    count: int,
    start_offset: int,
    forbidden: set[int] | None = None,
) -> list[int]:
    if forbidden is None:
        forbidden = set()
    candidates = [idx for idx, label in enumerate(labels.tolist()) if int(label) == support_label and idx not in forbidden]
    chosen = candidates[start_offset:start_offset + count]
    if len(chosen) < count:
        raise ValueError(
            f"not enough support samples for label {support_label}: need {count}, "
            f"have {max(0, len(candidates) - start_offset)} after offset"
        )
    return chosen


def score_metrics(scores: np.ndarray, target_label: int) -> dict[str, int]:
    target_score = int(scores[target_label])
    other_scores = np.delete(scores.astype(np.int64), target_label)
    max_other = int(np.max(other_scores)) if other_scores.size else 0
    pred = int(np.argmax(scores))
    return {
        "pred": pred,
        "target_score": target_score,
        "max_other_score": max_other,
        "margin": target_score - max_other,
        "hit": int(pred == target_label),
    }


def summarize(rows: list[dict]) -> dict:
    same_target_gains = np.array([row["same_target_gain"] for row in rows], dtype=np.float64)
    diff_target_gains = np.array([row["diff_target_gain"] for row in rows], dtype=np.float64)
    same_margin_gains = np.array([row["same_margin_gain"] for row in rows], dtype=np.float64)
    diff_margin_gains = np.array([row["diff_margin_gain"] for row in rows], dtype=np.float64)
    advantage = same_margin_gains - diff_margin_gains
    infer_hits = np.array([row["infer_hit"] for row in rows], dtype=np.float64)
    same_hits = np.array([row["same_hit"] for row in rows], dtype=np.float64)
    diff_hits = np.array([row["diff_hit"] for row in rows], dtype=np.float64)
    infer_stable = np.array([row["infer_stable"] for row in rows], dtype=np.float64)
    same_stable = np.array([row["same_stable"] for row in rows], dtype=np.float64)
    diff_stable = np.array([row["diff_stable"] for row in rows], dtype=np.float64)

    summary = {
        "num_cases": len(rows),
        "infer_hit_rate": float(np.mean(infer_hits)),
        "same_hit_rate": float(np.mean(same_hits)),
        "diff_hit_rate": float(np.mean(diff_hits)),
        "same_target_gain_mean": float(np.mean(same_target_gains)),
        "diff_target_gain_mean": float(np.mean(diff_target_gains)),
        "same_margin_gain_mean": float(np.mean(same_margin_gains)),
        "diff_margin_gain_mean": float(np.mean(diff_margin_gains)),
        "same_minus_diff_margin_gain_mean": float(np.mean(advantage)),
        "same_better_than_diff_fraction": float(np.mean(advantage > 0.0)),
        "same_positive_margin_gain_fraction": float(np.mean(same_margin_gains > 0.0)),
        "diff_positive_margin_gain_fraction": float(np.mean(diff_margin_gains > 0.0)),
        "infer_stable_fraction": float(np.mean(infer_stable)),
        "same_stable_fraction": float(np.mean(same_stable)),
        "diff_stable_fraction": float(np.mean(diff_stable)),
    }

    if (
        summary["same_minus_diff_margin_gain_mean"] > 0.0
        and summary["same_better_than_diff_fraction"] >= 0.6
        and summary["same_hit_rate"] >= summary["infer_hit_rate"]
    ):
        verdict = "evidence_for_stdp_effect"
        reason = "same-class support improved margin more often than different-class support"
    elif (
        summary["same_minus_diff_margin_gain_mean"] < 0.0
        and summary["same_hit_rate"] <= summary["diff_hit_rate"]
    ):
        verdict = "counter_evidence"
        reason = "different-class support was not worse than same-class support"
    else:
        verdict = "no_clear_evidence"
        reason = "the measured effect exists but is not strong or consistent enough yet"

    summary["verdict"] = verdict
    summary["reason"] = reason
    return summary


def save_benchmark(save_dir: Path, tag: str, metadata: dict, rows: list[dict], summary: dict) -> tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / f"{tag}_benchmark.json"
    csv_path = save_dir / f"{tag}_benchmark.csv"

    payload = {
        "metadata": metadata,
        "summary": summary,
        "cases": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fieldnames = [
        "test_index",
        "test_label",
        "same_support_label",
        "diff_support_label",
        "infer_pred",
        "same_pred",
        "diff_pred",
        "infer_hit",
        "same_hit",
        "diff_hit",
        "infer_target_score",
        "same_target_score",
        "diff_target_score",
        "infer_margin",
        "same_margin",
        "diff_margin",
        "same_target_gain",
        "diff_target_gain",
        "same_margin_gain",
        "diff_margin_gain",
        "same_latency_ms",
        "diff_latency_ms",
        "infer_latency_ms",
        "infer_stable",
        "same_stable",
        "diff_stable",
        "same_support_indices",
        "diff_support_indices",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return json_path, csv_path


def print_summary(summary: dict) -> None:
    print("\n========== STDP Effect Summary ==========")
    print(f"cases                         : {summary['num_cases']}")
    print(f"infer hit rate                : {summary['infer_hit_rate']:.3f}")
    print(f"same-class hit rate           : {summary['same_hit_rate']:.3f}")
    print(f"different-class hit rate      : {summary['diff_hit_rate']:.3f}")
    print(f"same target gain mean         : {summary['same_target_gain_mean']:.3f}")
    print(f"diff target gain mean         : {summary['diff_target_gain_mean']:.3f}")
    print(f"same margin gain mean         : {summary['same_margin_gain_mean']:.3f}")
    print(f"diff margin gain mean         : {summary['diff_margin_gain_mean']:.3f}")
    print(f"same-diff margin gain mean    : {summary['same_minus_diff_margin_gain_mean']:.3f}")
    print(f"same better than diff frac    : {summary['same_better_than_diff_fraction']:.3f}")
    print(f"same positive margin frac     : {summary['same_positive_margin_gain_fraction']:.3f}")
    print(f"diff positive margin frac     : {summary['diff_positive_margin_gain_fraction']:.3f}")
    print(f"infer stable frac             : {summary['infer_stable_fraction']:.3f}")
    print(f"same stable frac              : {summary['same_stable_fraction']:.3f}")
    print(f"diff stable frac              : {summary['diff_stable_fraction']:.3f}")
    print(f"verdict                       : {summary['verdict']}")
    print(f"reason                        : {summary['reason']}")
    print("=========================================\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark whether same-class STDP support helps more than different-class support"
    )
    parser.add_argument("--bit", type=Path, default=None, help="Path to .bit file")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory with MNIST raw files")
    parser.add_argument("--support-split", choices=["train", "test"], default="train")
    parser.add_argument("--test-split", choices=["train", "test"], default="test")
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--test-start-offset", type=int, default=0)
    parser.add_argument("--support-start-offset", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--dma-ip", default="axi_dma_0")
    parser.add_argument("--save-dir", type=Path, default=Path("./logs"))
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--diff-label-mode",
        choices=["next", "fixed"],
        default="next",
        help="How to choose the different-class support label",
    )
    parser.add_argument("--diff-label", type=int, default=None, help="Used when --diff-label-mode fixed")
    args = parser.parse_args()

    if args.repeat <= 0:
        raise ValueError("repeat must be positive")
    if args.diff_label is not None and not (0 <= args.diff_label <= 9):
        raise ValueError("--diff-label must be in range [0, 9]")

    bit_path = resolve_existing_path(args.bit, default_bit_candidates(), "bitstream")
    hwh_path = bit_path.with_suffix(".hwh")
    if not hwh_path.exists():
        raise FileNotFoundError(f"hwh not found next to bitstream: {hwh_path}")

    raw_dir = resolve_existing_path(args.raw_dir, default_raw_dir_candidates(), "MNIST raw directory")
    support_images_all, support_labels_all = load_split(raw_dir, args.support_split)
    test_images_all, test_labels_all = load_split(raw_dir, args.test_split)

    test_case_indices = select_test_cases(test_labels_all, args.samples_per_class, args.test_start_offset)

    from pynq import Overlay, allocate

    overlay = Overlay(str(bit_path))
    dma = resolve_dma(overlay, args.dma_ip)
    init_dma_channels(dma)

    rows: list[dict] = []

    print("\n========== STDP Effect Benchmark ==========")
    print(f"Overlay bit        : {bit_path}")
    print(f"Overlay hwh        : {hwh_path}")
    print(f"Support split      : {args.support_split}")
    print(f"Test split         : {args.test_split}")
    print(f"Samples per class  : {args.samples_per_class}")
    print(f"Support count      : {TRAIN_IMAGE_COUNT}")
    print(f"DMA IP             : {args.dma_ip}")
    print("===========================================\n")

    for test_index in test_case_indices:
        test_label = int(test_labels_all[test_index])
        test_image = test_images_all[test_index].reshape(-1).astype(np.uint8)

        if args.diff_label_mode == "fixed":
            if args.diff_label is None:
                raise ValueError("--diff-label must be provided when --diff-label-mode fixed")
            diff_label = int(args.diff_label)
            if diff_label == test_label:
                raise ValueError("fixed different-class label must differ from the test label")
        else:
            diff_label = (test_label + 1) % 10

        forbidden: set[int] = set()
        if args.support_split == args.test_split:
            forbidden.add(test_index)

        same_support_indices = choose_support_indices(
            support_labels_all,
            support_label=test_label,
            count=TRAIN_IMAGE_COUNT,
            start_offset=args.support_start_offset,
            forbidden=forbidden,
        )
        diff_support_indices = choose_support_indices(
            support_labels_all,
            support_label=diff_label,
            count=TRAIN_IMAGE_COUNT,
            start_offset=args.support_start_offset,
            forbidden=forbidden,
        )

        same_support_images = [
            support_images_all[idx].reshape(-1).astype(np.uint8) for idx in same_support_indices
        ]
        diff_support_images = [
            support_images_all[idx].reshape(-1).astype(np.uint8) for idx in diff_support_indices
        ]

        infer_scores, infer_ms, infer_stable = run_mode(
            dma=dma,
            allocate_fn=allocate,
            mode=MODE_INFER,
            test_image=test_image,
            train_images=[],
            repeat=args.repeat,
            timeout_s=args.timeout_s,
        )
        same_scores, same_ms, same_stable = run_mode(
            dma=dma,
            allocate_fn=allocate,
            mode=MODE_TRAIN,
            test_image=test_image,
            train_images=same_support_images,
            repeat=args.repeat,
            timeout_s=args.timeout_s,
        )
        diff_scores, diff_ms, diff_stable = run_mode(
            dma=dma,
            allocate_fn=allocate,
            mode=MODE_TRAIN,
            test_image=test_image,
            train_images=diff_support_images,
            repeat=args.repeat,
            timeout_s=args.timeout_s,
        )

        infer_m = score_metrics(infer_scores, test_label)
        same_m = score_metrics(same_scores, test_label)
        diff_m = score_metrics(diff_scores, test_label)

        row = {
            "test_index": test_index,
            "test_label": test_label,
            "same_support_label": test_label,
            "diff_support_label": diff_label,
            "same_support_indices": ",".join(map(str, same_support_indices)),
            "diff_support_indices": ",".join(map(str, diff_support_indices)),
            "infer_pred": infer_m["pred"],
            "same_pred": same_m["pred"],
            "diff_pred": diff_m["pred"],
            "infer_hit": infer_m["hit"],
            "same_hit": same_m["hit"],
            "diff_hit": diff_m["hit"],
            "infer_target_score": infer_m["target_score"],
            "same_target_score": same_m["target_score"],
            "diff_target_score": diff_m["target_score"],
            "infer_margin": infer_m["margin"],
            "same_margin": same_m["margin"],
            "diff_margin": diff_m["margin"],
            "same_target_gain": same_m["target_score"] - infer_m["target_score"],
            "diff_target_gain": diff_m["target_score"] - infer_m["target_score"],
            "same_margin_gain": same_m["margin"] - infer_m["margin"],
            "diff_margin_gain": diff_m["margin"] - infer_m["margin"],
            "infer_latency_ms": round(infer_ms, 3),
            "same_latency_ms": round(same_ms, 3),
            "diff_latency_ms": round(diff_ms, 3),
            "infer_stable": int(infer_stable),
            "same_stable": int(same_stable),
            "diff_stable": int(diff_stable),
        }
        rows.append(row)

        print(
            f"test={test_index:4d} label={test_label} | "
            f"infer(pred={row['infer_pred']}, margin={row['infer_margin']}) | "
            f"same(pred={row['same_pred']}, d_margin={row['same_margin_gain']:+d}) | "
            f"diff(pred={row['diff_pred']}, d_margin={row['diff_margin_gain']:+d})"
        )

    summary = summarize(rows)
    print_summary(summary)

    tag = args.tag or datetime.now().strftime("stdp_effect_%Y%m%d_%H%M%S")
    metadata = {
        "overlay_bit": str(bit_path),
        "overlay_hwh": str(hwh_path),
        "raw_dir": str(raw_dir),
        "support_split": args.support_split,
        "test_split": args.test_split,
        "samples_per_class": args.samples_per_class,
        "test_start_offset": args.test_start_offset,
        "support_start_offset": args.support_start_offset,
        "support_count": TRAIN_IMAGE_COUNT,
        "repeat": args.repeat,
        "timeout_s": args.timeout_s,
        "dma_ip": args.dma_ip,
        "diff_label_mode": args.diff_label_mode,
        "diff_label": args.diff_label,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    json_path, csv_path = save_benchmark(args.save_dir, tag, metadata, rows, summary)
    print(f"Saved benchmark json: {json_path}")
    print(f"Saved benchmark csv : {csv_path}")

    if summary["verdict"] == "evidence_for_stdp_effect":
        return 0
    if summary["verdict"] == "no_clear_evidence":
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
