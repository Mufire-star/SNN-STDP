#!/usr/bin/env python3

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path


FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://storage.googleapis.com/cvdf-datasets/mnist",
]


def download_one(gz_name: str, out_dir: Path, keep_gz: bool) -> None:
    raw_name = gz_name[:-3]
    raw_path = out_dir / raw_name
    gz_path = out_dir / gz_name

    if raw_path.exists():
        print(f"skip {raw_name}: already present")
        return

    last_error = None
    for base in MIRRORS:
        url = f"{base}/{gz_name}"
        try:
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, gz_path)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    else:
        raise RuntimeError(f"failed to download {gz_name}: {last_error}") from last_error

    with gzip.open(gz_path, "rb") as src, raw_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)

    if not keep_gz:
        gz_path.unlink(missing_ok=True)

    print(f"ready {raw_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MNIST raw files")
    parser.add_argument("--out", type=Path, default=Path("data/mnist/raw"))
    parser.add_argument("--keep-gz", action="store_true", help="keep downloaded .gz files")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download_one(name, args.out, args.keep_gz)

    print(f"MNIST raw dataset ready at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
