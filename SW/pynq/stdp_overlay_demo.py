#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path


MODE_INFER = 0
MODE_TRAIN = 1
IMG_H = 28
IMG_W = 28
IMG_BYTES = IMG_H * IMG_W
OUTPUT_CLASSES = 10
OUTPUT_BYTES = OUTPUT_CLASSES * 2
NUM_TRAIN_IMG = 10
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_GAINS = (1.0, 0.75, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08)

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
    script = script_dir()
    if script == Path.cwd():
        return Path.cwd()
    return script.parents[1]


def script_dir() -> Path:
    file_name = globals().get("__file__")
    if not file_name:
        return Path.cwd()
    return Path(file_name).resolve().parent


def default_bit_candidates() -> list[Path]:
    root = repo_root()
    return [
        Path.cwd() / "snn_stdp.bit",
        script_dir() / "snn_stdp.bit",
        root / "HW/stdp_snn/vivado/output_stdp/snn_stdp.bit",
    ]


def default_image_candidates() -> list[Path]:
    root = repo_root()
    search_dirs = [Path.cwd(), script_dir(), root]
    candidates: list[Path] = []

    for directory in search_dirs:
        candidates.extend([directory / f"{label}.jpg" for label in range(10)])
        candidates.extend(image_files_in_dir(directory))

    return unique_paths(candidates)


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        normalized = path.resolve() if path.exists() else path.absolute()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    return unique


def image_files_in_dir(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def resolve_existing_path(path: Path | None, candidates: list[Path], description: str) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"{description} not found: {path}")
        return path

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"could not find {description}; tried:\n{tried}")


def resolve_image_path(path: Path | None) -> Path:
    if path is not None:
        if path.is_dir():
            images = image_files_in_dir(path)
            if not images:
                exts = ", ".join(IMAGE_EXTENSIONS)
                raise FileNotFoundError(f"no image files ({exts}) found in directory: {path}")
            return images[0]
        if not path.exists():
            raise FileNotFoundError(f"input image not found: {path}")
        return path

    return resolve_existing_path(None, default_image_candidates(), "input image")


def resolve_batch_image_paths(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"batch directory not found: {directory}")

    image_paths = [directory / f"{label}.jpg" for label in range(10)]
    missing = [path.name for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing batch images in {directory}: {', '.join(missing)}"
        )
    return image_paths


def label_from_filename(path: Path) -> int | None:
    stem = path.stem
    if stem.isdigit() and len(stem) == 1:
        return int(stem)
    if stem and stem[0].isdigit():
        return int(stem[0])
    return None


def resolve_support_paths(image_path: Path, support_dir: Path | None) -> list[tuple[int, Path]]:
    directory = support_dir if support_dir is not None else image_path.parent
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"support directory not found: {directory}")

    images = image_files_in_dir(directory)
    by_label: dict[int, Path] = {}
    for path in images:
        label = label_from_filename(path)
        if label is None or not (0 <= label < OUTPUT_CLASSES):
            continue
        by_label.setdefault(label, path)

    missing = [label for label in range(NUM_TRAIN_IMG) if label not in by_label]
    if missing:
        labels = ", ".join(str(label) for label in missing)
        raise FileNotFoundError(
            f"missing support images for labels {labels} in {directory}; "
            "put 0.jpg ... 9.jpg next to the test image or pass --mode infer"
        )

    return [(label, by_label[label]) for label in range(NUM_TRAIN_IMG)]


def load_image(
    path: Path,
    invert: bool,
    raw_resize: bool,
    threshold: int | None,
    auto_invert: bool,
) -> np.ndarray:
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("L")
    if raw_resize:
        pixels = np.array(image.resize((IMG_W, IMG_H)), dtype=np.uint8)
        if invert:
            pixels = np.uint8(255) - pixels
        return pixels.reshape(IMG_BYTES)

    pixels = np.array(image, dtype=np.uint8)
    if auto_invert:
        pixels = normalize_foreground_bright(pixels)
    if invert:
        pixels = np.uint8(255) - pixels
    return crop_center_resize(pixels, threshold).reshape(IMG_BYTES)


def normalize_foreground_bright(pixels: np.ndarray) -> np.ndarray:
    import numpy as np

    border = np.concatenate(
        [pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]]
    )
    if float(np.median(border)) > 127.0:
        return np.uint8(255) - pixels
    return pixels


def auto_threshold(pixels: np.ndarray) -> int:
    import numpy as np

    hist = np.bincount(pixels.reshape(-1), minlength=256).astype(np.float64)
    total = pixels.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_bg = 0.0
    weight_bg = 0.0
    best_t = 0
    best_var = -1.0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best_var:
            best_var = between
            best_t = t

    return int(best_t)


def crop_center_resize(pixels: np.ndarray, threshold: int | None) -> np.ndarray:
    import numpy as np
    from PIL import Image

    t = auto_threshold(pixels) if threshold is None else threshold
    mask = pixels > t
    if not np.any(mask):
        return np.array(Image.fromarray(pixels).resize((IMG_W, IMG_H)), dtype=np.uint8)

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad = max(2, int(max(y1 - y0 + 1, x1 - x0 + 1) * 0.08))
    y0 = max(0, y0 - pad)
    y1 = min(pixels.shape[0] - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(pixels.shape[1] - 1, x1 + pad)

    crop = pixels[y0 : y1 + 1, x0 : x1 + 1]
    scale = min(20 / crop.shape[1], 20 / crop.shape[0])
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_h = max(1, int(round(crop.shape[0] * scale)))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    resized = np.array(
        Image.fromarray(crop).resize((new_w, new_h), resampling),
        dtype=np.uint8,
    )

    canvas = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    y = (IMG_H - new_h) // 2
    x = (IMG_W - new_w) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def scale_image(image: np.ndarray, gain: float) -> np.ndarray:
    import numpy as np

    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def image_stats(image: np.ndarray, gain: float = 1.0) -> dict[str, float]:
    import numpy as np

    values = np.clip(image.astype(np.float32) * gain, 0, 255)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "nonzero_pct": float(np.mean(values > 0) * 100.0),
        "active_pct": float(np.mean(values > 32) * 100.0),
    }


def scores_are_flat(scores: np.ndarray) -> bool:
    import numpy as np

    return bool(np.all(scores == scores[0]))


def score_spread(scores: np.ndarray) -> int:
    import numpy as np

    return int(np.max(scores.astype(np.int64)) - np.min(scores.astype(np.int64)))


def scores_are_saturated(scores: np.ndarray) -> bool:
    return scores_are_flat(scores) and int(scores[0]) >= 256


def ascii_image(image: np.ndarray, gain: float) -> str:
    import numpy as np

    values = np.clip(image.reshape(IMG_H, IMG_W).astype(np.float32) * gain, 0, 255)
    chars = " .:-=+*#%@"
    lines = []
    for row in values:
        line = "".join(chars[min(len(chars) - 1, int(v * len(chars) / 256))] for v in row)
        lines.append(line.rstrip())
    return "\n".join(lines)


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


def resolve_dma(overlay, dma_name: str):
    if dma_name in overlay.ip_dict:
        return getattr(overlay, dma_name)

    available = ", ".join(sorted(overlay.ip_dict.keys()))
    raise KeyError(f"{dma_name} not found in overlay; available IPs: {available}")


def build_payload(image: np.ndarray) -> np.ndarray:
    import numpy as np

    if image.shape != (IMG_BYTES,):
        raise ValueError(f"image must contain {IMG_BYTES} pixels, got shape {image.shape}")

    payload = np.empty((1 + IMG_BYTES,), dtype=np.uint8)
    payload[0] = MODE_INFER
    payload[1:] = image
    return payload


def build_train_payload(
    support_images: list[tuple[int, np.ndarray]],
    test_image: np.ndarray,
) -> np.ndarray:
    import numpy as np

    if len(support_images) != NUM_TRAIN_IMG:
        raise ValueError(f"expected {NUM_TRAIN_IMG} support images, got {len(support_images)}")
    if test_image.shape != (IMG_BYTES,):
        raise ValueError(f"test image must contain {IMG_BYTES} pixels, got shape {test_image.shape}")

    payload = np.empty((1 + NUM_TRAIN_IMG * (1 + IMG_BYTES) + IMG_BYTES,), dtype=np.uint8)
    cursor = 0
    payload[cursor] = MODE_TRAIN
    cursor += 1
    for label, image in support_images:
        if image.shape != (IMG_BYTES,):
            raise ValueError(f"support image must contain {IMG_BYTES} pixels, got shape {image.shape}")
        if not (0 <= label < OUTPUT_CLASSES):
            raise ValueError(f"support label out of range: {label}")
        payload[cursor] = label
        cursor += 1
        payload[cursor : cursor + IMG_BYTES] = image
        cursor += IMG_BYTES
    payload[cursor : cursor + IMG_BYTES] = test_image
    return payload


def run_payload(dma, allocate_fn, payload: np.ndarray, timeout_s: float) -> np.ndarray:
    import numpy as np

    in_buf = allocate_fn(shape=(payload.size,), dtype=np.uint8)
    out_buf = allocate_fn(shape=(OUTPUT_CLASSES,), dtype=np.uint16)
    try:
        in_buf[:] = payload
        out_buf[:] = 0
        in_buf.flush()
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
        scores = np.array(out_buf, copy=True)
    finally:
        in_buf.freebuffer()
        out_buf.freebuffer()

    validate_scores(scores)
    return scores


def run_once(dma, allocate_fn, image: np.ndarray, timeout_s: float) -> np.ndarray:
    return run_payload(dma, allocate_fn, build_payload(image), timeout_s)


def validate_scores(scores: np.ndarray) -> None:
    if scores.shape != (OUTPUT_CLASSES,):
        raise RuntimeError(f"invalid score shape {scores.shape}, expected {(OUTPUT_CLASSES,)}")


def topk_pairs(scores: np.ndarray, topk: int = 3) -> list[tuple[int, int]]:
    import numpy as np

    order = np.argsort(scores)[::-1][:topk]
    return [(int(idx), int(scores[idx])) for idx in order]


def run_single_image(
    bit_path: Path,
    image_path: Path,
    mode: str,
    support_dir: Path | None,
    dma_name: str,
    timeout_s: float,
    invert: bool,
    raw_resize: bool,
    threshold: int | None,
    auto_invert: bool,
    gain: float | None,
    auto_gain: bool,
) -> tuple[np.ndarray, float, float, dict[str, float], np.ndarray, list[tuple[int, Path]]]:
    from pynq import Overlay, allocate

    hwh_path = bit_path.with_suffix(".hwh")
    if not hwh_path.exists():
        raise FileNotFoundError(f"hwh not found next to bitstream: {hwh_path}")

    image = load_image(
        image_path,
        invert=invert,
        raw_resize=raw_resize,
        threshold=threshold,
        auto_invert=auto_invert,
    )
    support_paths = resolve_support_paths(image_path, support_dir) if mode == "train" else []
    support_images = [
        (
            label,
            load_image(
                path,
                invert=invert,
                raw_resize=raw_resize,
                threshold=threshold,
                auto_invert=auto_invert,
            ),
        )
        for label, path in support_paths
    ]

    overlay = Overlay(str(bit_path))
    dma = resolve_dma(overlay, dma_name)
    init_dma_channels(dma)

    if gain is not None:
        gains = [gain]
    elif mode == "train" or not auto_gain:
        gains = [1.0]
    else:
        gains = list(DEFAULT_GAINS)
    attempts: list[tuple[float, np.ndarray, float]] = []

    for current_gain in gains:
        test_image = scale_image(image, current_gain)
        if mode == "train":
            scaled_support = [
                (label, scale_image(support_image, current_gain))
                for label, support_image in support_images
            ]
            payload = build_train_payload(scaled_support, test_image)
        else:
            payload = build_payload(test_image)

        t0 = time.perf_counter()
        scores = run_payload(dma, allocate, payload, timeout_s=timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1e3
        attempts.append((current_gain, scores, latency_ms))

        if gain is not None or not auto_gain:
            break

    nonflat = [attempt for attempt in attempts if score_spread(attempt[1]) > 0]
    if nonflat:
        best_gain, best_scores, best_latency_ms = max(nonflat, key=lambda item: score_spread(item[1]))
    else:
        nonsaturated = [attempt for attempt in attempts if not scores_are_saturated(attempt[1])]
        best_gain, best_scores, best_latency_ms = nonsaturated[0] if nonsaturated else attempts[-1]

    stats = image_stats(image, best_gain)
    return best_scores, best_latency_ms, best_gain, stats, image, support_paths


def print_result(
    bit_path: Path,
    image_path: Path,
    mode: str,
    scores: np.ndarray,
    latency_ms: float,
    gain: float,
    stats: dict[str, float],
    image: np.ndarray,
    support_paths: list[tuple[int, Path]],
    show_pixels: bool,
) -> None:
    import numpy as np

    pred = int(np.argmax(scores))
    title = "Train-Then-Infer" if mode == "train" else "Single Image Inference"
    print(f"\n========== {title} ==========")
    print(f"Bitstream : {bit_path}")
    print(f"Image     : {image_path}")
    if support_paths:
        support_desc = ", ".join(f"{label}:{path.name}" for label, path in support_paths)
        print(f"Support   : {support_desc}")
    print(
        "Input     : "
        f"min={stats['min']:.0f}, max={stats['max']:.0f}, mean={stats['mean']:.1f}, "
        f"nonzero={stats['nonzero_pct']:.1f}%, active={stats['active_pct']:.1f}%, gain={gain:g}"
    )
    print(f"Scores    : {scores.astype(np.uint16).tolist()}")
    print(f"Top3      : {topk_pairs(scores)}")
    print(f"Predicted : {pred}")
    print(f"Latency   : {latency_ms:.3f} ms")
    if scores_are_flat(scores):
        print("Warning   : scores are flat; this hardware/input is saturated or not class-selective")
        if scores_are_saturated(scores):
            print("Hint      : try a cleaner black-background digit, or retrain with a better support set")
    if show_pixels:
        print("Pixels:")
        print(ascii_image(image, gain))
    print("============================================\n")


def print_batch_result(
    image_path: Path,
    scores: np.ndarray,
    latency_ms: float,
    expected: int | None,
) -> None:
    import numpy as np

    pred = int(np.argmax(scores))
    expected_text = f", expected={expected}" if expected is not None else ""
    print(
        f"{image_path.name}: pred={pred}{expected_text}, "
        f"latency={latency_ms:.3f} ms, top3={topk_pairs(scores)}, "
        f"scores={scores.astype(np.uint16).tolist()}"
    )


def strip_ipykernel_args(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False

    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "-f":
            skip_next = True
            continue
        if arg.startswith("-f=") or arg.startswith("--f="):
            continue
        cleaned.append(arg)

    return cleaned


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        import sys

        argv = strip_ipykernel_args(sys.argv[1:])

    parser = argparse.ArgumentParser(description="Run one image through the STDP SNN overlay")
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=None,
        help="Path to one image file, or a directory where the first image will be used",
    )
    parser.add_argument("--bit", type=Path, default=None, help="Path to snn_stdp.bit")
    parser.add_argument(
        "--mode",
        choices=("train", "infer"),
        default="train",
        help="train runs label-gated STDP on 0.jpg ... 9.jpg before testing the image",
    )
    parser.add_argument(
        "--support-dir",
        type=Path,
        default=None,
        help="Directory containing labeled support images named 0.jpg ... 9.jpg",
    )
    parser.add_argument("--dma-ip", default="axi_dma_0")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--invert", action="store_true", help="Invert grayscale pixels before inference")
    parser.add_argument(
        "--no-auto-invert",
        action="store_true",
        help="Disable automatic white-background inversion",
    )
    parser.add_argument(
        "--raw-resize",
        action="store_true",
        help="Only resize to 28x28; skip MNIST-style crop/center preprocessing",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Foreground threshold after inversion; default uses Otsu threshold",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Scale input pixel intensity before DMA; disables automatic gain sweep",
    )
    parser.add_argument(
        "--no-auto-gain",
        action="store_true",
        help="Do not retry with lower pixel gains when all scores saturate at 256",
    )
    parser.add_argument(
        "--show-pixels",
        action="store_true",
        help="Print the preprocessed 28x28 input as ASCII pixels",
    )
    args = parser.parse_args(argv)
    if args.threshold is not None and not (0 <= args.threshold <= 255):
        raise ValueError("--threshold must be in range [0, 255]")
    if args.gain is not None and args.gain <= 0:
        raise ValueError("--gain must be positive")

    bit_path = resolve_existing_path(args.bit, default_bit_candidates(), "bitstream")
    image_path = resolve_image_path(args.image)
    scores, latency_ms, used_gain, stats, processed_image, support_paths = run_single_image(
        bit_path=bit_path,
        image_path=image_path,
        mode=args.mode,
        support_dir=args.support_dir,
        dma_name=args.dma_ip,
        timeout_s=args.timeout_s,
        invert=args.invert,
        raw_resize=args.raw_resize,
        threshold=args.threshold,
        auto_invert=not args.no_auto_invert,
        gain=args.gain,
        auto_gain=not args.no_auto_gain,
    )
    print_result(
        bit_path,
        image_path,
        args.mode,
        scores,
        latency_ms,
        used_gain,
        stats,
        processed_image,
        support_paths,
        args.show_pixels,
    )
    return 0


if __name__ == "__main__":
    main()
