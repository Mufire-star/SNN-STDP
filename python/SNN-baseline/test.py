import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from spikingjelly.activation_based import encoding

# 复用训练脚本里的数据集/模型/评估与加载逻辑
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train as snn_train


def main() -> int:
    parser = argparse.ArgumentParser(description='Test the trained SNN model (MNIST).')
    parser.add_argument('--data-dir', type=str, default=str(Path(__file__).resolve().parent / 'dates'))
    parser.add_argument('--ckpt', type=str, default=str(Path(__file__).resolve().parent / 'mnist_snn.pt'))
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--T', type=int, default=0, help='override time steps; 0 means use checkpoint/default')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-samples', type=int, default=5000, help='test first N samples from test set (0 means all)')
    parser.add_argument('--print-each', action='store_true', help='print index/pred/label for each tested sample')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.ckpt)

    # 准备数据
    test_ds = snn_train.MNISTIdxDataset(
        data_dir / 't10k-images.idx3-ubyte',
        data_dir / 't10k-labels.idx1-ubyte',
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith('cuda') and torch.cuda.is_available()),
    )

    # 先读 checkpoint，拿到 channels / T（如果你训练时改过）
    if not ckpt_path.exists():
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location=args.device)
    ckpt_channels = int(ckpt.get('channels', 16))
    ckpt_T = int(ckpt.get('T', 8))

    T = int(args.T) if int(args.T) > 0 else ckpt_T

    # 构建模型并加载参数
    model = snn_train.MNISTConvSNN(channels=ckpt_channels, step_mode='m').to(args.device)
    model.load_state_dict(ckpt['state_dict'], strict=True)

    encoder = encoding.PoissonEncoder()

    cfg = snn_train.TrainConfig(
        T=T,
        channels=ckpt_channels,
        batch_size=args.batch_size,
        epochs=1,
        lr=0.0,
        num_workers=args.num_workers,
        device=args.device,
    )

    # 评估（可限制前 N 个样本）
    model.eval()
    correct = 0
    total = 0
    max_samples = int(args.max_samples)
    with torch.no_grad():
        for x, y in test_loader:
            if max_samples > 0 and total >= max_samples:
                break

            x = x.to(args.device)
            y = y.to(args.device)
            snn_train.functional.reset_net(model)

            x_seq = torch.stack([encoder(x) for _ in range(T)], dim=0)
            y_seq = model(x_seq)
            fr = y_seq.mean(0)
            pred = fr.argmax(1)

            for i in range(pred.numel()):
                if max_samples > 0 and total >= max_samples:
                    break
                if args.print_each:
                    print(f'idx={total:05d} pred={int(pred[i])} label={int(y[i])}')
                correct += int((pred[i] == y[i]).item())
                total += 1

    acc = correct / max(total, 1)
    print(f'ckpt={ckpt_path}')
    print(f'loaded: epoch={ckpt.get("epoch")} acc={ckpt.get("acc")} channels={ckpt_channels} T={T}')
    print(f'tested_samples={total} correct={correct} acc={acc:.4f}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
