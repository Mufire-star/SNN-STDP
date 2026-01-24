import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Union

from spikingjelly.activation_based import encoding, functional, layer, neuron, surrogate


def read_idx_images(path: Path) -> np.ndarray:
	with path.open('rb') as f:
		header = f.read(16)
		magic, num, rows, cols = struct.unpack('>IIII', header)
		if magic != 2051:
			raise ValueError(f'Invalid IDX3 magic={magic} for {path}')
		data = np.frombuffer(f.read(), dtype=np.uint8)
	return data.reshape(num, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
	with path.open('rb') as f:
		header = f.read(8)
		magic, num = struct.unpack('>II', header)
		if magic != 2049:
			raise ValueError(f'Invalid IDX1 magic={magic} for {path}')
		data = np.frombuffer(f.read(), dtype=np.uint8)
	return data.reshape(num)


class MNISTIdxDataset(Dataset):
	def __init__(self, images_path: Path, labels_path: Path):
		self.images = read_idx_images(images_path)
		self.labels = read_idx_labels(labels_path)
		if self.images.shape[0] != self.labels.shape[0]:
			raise ValueError('images and labels count mismatch')

	def __len__(self) -> int:
		return int(self.labels.shape[0])

	def __getitem__(self, index: int):
		x = self.images[index].astype(np.float32) / 255.0  # [H, W] in [0, 1]
		x = torch.from_numpy(x).unsqueeze(0)  # [1, 28, 28]
		y = int(self.labels[index])
		return x, y


class MNISTConvSNN(nn.Module):
	def __init__(self, channels: int = 16, step_mode: str = 'm'):
		super().__init__()
		self.net = nn.Sequential(
			layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False, step_mode=step_mode),
			layer.BatchNorm2d(channels, step_mode=step_mode),
			neuron.LIFNode(
				tau=2.0,
				surrogate_function=surrogate.ATan(),
				detach_reset=True,
				step_mode=step_mode,
			),
			layer.MaxPool2d(kernel_size=2, stride=2, step_mode=step_mode),

			layer.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False, step_mode=step_mode),
			layer.BatchNorm2d(channels * 2, step_mode=step_mode),
			neuron.LIFNode(
				tau=2.0,
				surrogate_function=surrogate.ATan(),
				detach_reset=True,
				step_mode=step_mode,
			),
			layer.MaxPool2d(kernel_size=2, stride=2, step_mode=step_mode),

			layer.Flatten(step_mode=step_mode),
			layer.Linear((channels * 2) * 7 * 7, 10, step_mode=step_mode),
			neuron.LIFNode(
				tau=2.0,
				surrogate_function=surrogate.ATan(),
				detach_reset=True,
				step_mode=step_mode,
			),
		)

	def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
		# x_seq: [T, N, 1, 28, 28]
		return self.net(x_seq)


def save_checkpoint(path: Path, model: nn.Module, cfg: 'TrainConfig', epoch: int, acc: float) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	ckpt = {
		'epoch': int(epoch),
		'acc': float(acc),
		'T': int(cfg.T),
		'channels': int(cfg.channels),
		'state_dict': model.state_dict(),
	}
	torch.save(ckpt, path)


def load_checkpoint(path: Path, model: nn.Module, map_location: Union[str, torch.device] = 'cpu') -> dict:
	ckpt = torch.load(path, map_location=map_location)
	state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
	model.load_state_dict(state_dict, strict=True)
	return ckpt


def save_full_checkpoint(path: Path, model: nn.Module, opt, scheduler, cfg: 'TrainConfig', epoch: int, acc: float, best_acc: float) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	ckpt = {
		'epoch': int(epoch),
		'acc': float(acc),
		'best_acc': float(best_acc),
		'T': int(cfg.T),
		'channels': int(cfg.channels),
		'state_dict': model.state_dict(),
		'optimizer': opt.state_dict() if opt is not None else None,
		'scheduler': scheduler.state_dict() if scheduler is not None else None,
	}
	torch.save(ckpt, path)


@dataclass
class TrainConfig:
	T: int = 8
	channels: int = 16
	batch_size: int = 64
	epochs: int = 10
	lr: float = 1e-3
	num_workers: int = 0
	device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def evaluate(model: nn.Module, encoder: encoding.PoissonEncoder, loader: DataLoader, cfg: TrainConfig) -> float:
	model.eval()
	correct = 0
	total = 0
	for x, y in loader:
		x = x.to(cfg.device, non_blocking=True)
		y = y.to(cfg.device, non_blocking=True)
		functional.reset_net(model)

		x_seq = torch.stack([encoder(x) for _ in range(cfg.T)], dim=0)
		y_seq = model(x_seq)
		fr = y_seq.mean(0)  # [N, 10]
		pred = fr.argmax(1)
		correct += int((pred == y).sum().item())
		total += int(y.numel())
	return correct / max(total, 1)


def train_one_epoch(model: nn.Module, encoder: encoding.PoissonEncoder, loader: DataLoader, opt, cfg: TrainConfig) -> float:
	model.train()
	total_loss = 0.0
	num_batches = 0
	for x, y in loader:
		x = x.to(cfg.device, non_blocking=True)
		y = y.to(cfg.device, non_blocking=True)

		functional.reset_net(model)
		x_seq = torch.stack([encoder(x) for _ in range(cfg.T)], dim=0)
		y_seq = model(x_seq)
		fr = y_seq.mean(0)

		loss = F.cross_entropy(fr, y)
		opt.zero_grad(set_to_none=True)
		loss.backward()
		opt.step()

		total_loss += float(loss.item())
		num_batches += 1
	return total_loss / max(num_batches, 1)


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', type=str, default=str(Path(__file__).resolve().parent / 'dates'))
	parser.add_argument('--T', type=int, default=8)
	parser.add_argument('--channels', type=int, default=16)
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--weight-decay', type=float, default=0.0)
	parser.add_argument('--seed', type=int, default=123)
	parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'cosine', 'step'])
	parser.add_argument('--step-size', type=int, default=50)
	parser.add_argument('--gamma', type=float, default=0.5)
	parser.add_argument('--save', type=str, default=str(Path(__file__).resolve().parent / 'mnist_snn.pt'))
	parser.add_argument('--load', type=str, default='')
	parser.add_argument('--eval-only', action='store_true')
	parser.add_argument('--resume', action='store_true', help='when used with --load, also resume optimizer/scheduler and epoch')
	parser.add_argument('--save-every', type=int, default=50, help='save a periodic checkpoint every N epochs (0 disables)')
	args = parser.parse_args()

	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	cfg = TrainConfig(T=args.T, channels=args.channels, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
	data_dir = Path(args.data_dir)

	train_ds = MNISTIdxDataset(data_dir / 'train-images.idx3-ubyte', data_dir / 'train-labels.idx1-ubyte')
	test_ds = MNISTIdxDataset(data_dir / 't10k-images.idx3-ubyte', data_dir / 't10k-labels.idx1-ubyte')

	train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
	test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

	model = MNISTConvSNN(channels=cfg.channels, step_mode='m').to(cfg.device)
	encoder = encoding.PoissonEncoder()
	opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=float(args.weight_decay))
	if args.scheduler == 'cosine':
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
	elif args.scheduler == 'step':
		scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(int(args.step_size), 1), gamma=float(args.gamma))
	else:
		scheduler = None

	if args.load:
		ckpt = torch.load(Path(args.load), map_location=cfg.device)
		if isinstance(ckpt, dict) and 'state_dict' in ckpt:
			model.load_state_dict(ckpt['state_dict'], strict=True)
		else:
			model.load_state_dict(ckpt, strict=True)
		print(f"loaded ckpt: epoch={ckpt.get('epoch') if isinstance(ckpt, dict) else None} acc={ckpt.get('acc') if isinstance(ckpt, dict) else None} T={ckpt.get('T') if isinstance(ckpt, dict) else None} channels={ckpt.get('channels') if isinstance(ckpt, dict) else None}")
		if args.eval_only:
			acc = evaluate(model, encoder, test_loader, cfg)
			print(f'eval acc={acc:.4f}')
			return 0
		if args.resume and isinstance(ckpt, dict):
			if ckpt.get('optimizer') is not None:
				opt.load_state_dict(ckpt['optimizer'])
			if scheduler is not None and ckpt.get('scheduler') is not None:
				scheduler.load_state_dict(ckpt['scheduler'])
			print('resumed optimizer/scheduler state')

	best_acc = float(ckpt.get('best_acc', -1.0)) if (args.load and args.resume and isinstance(ckpt, dict)) else -1.0
	save_path = Path(args.save)
	start_epoch = int(ckpt.get('epoch', 0)) + 1 if (args.load and args.resume and isinstance(ckpt, dict)) else 1
	end_epoch = max(start_epoch, cfg.epochs)

	for epoch in range(start_epoch, end_epoch + 1):
		loss = train_one_epoch(model, encoder, train_loader, opt, cfg)
		acc = evaluate(model, encoder, test_loader, cfg)
		print(f'epoch={epoch} loss={loss:.4f} acc={acc:.4f}')
		if scheduler is not None:
			scheduler.step()
		if acc > best_acc:
			best_acc = acc
			save_full_checkpoint(save_path, model, opt, scheduler, cfg, epoch=epoch, acc=acc, best_acc=best_acc)
			print(f'saved ckpt: {save_path} (best_acc={best_acc:.4f})')
		elif args.save_every and (epoch % int(args.save_every) == 0):
			periodic = save_path.with_name(save_path.stem + f'_e{epoch}' + save_path.suffix)
			save_full_checkpoint(periodic, model, opt, scheduler, cfg, epoch=epoch, acc=acc, best_acc=best_acc)
			print(f'saved periodic ckpt: {periodic}')

	return 0


if __name__ == '__main__':
	raise SystemExit(main())
