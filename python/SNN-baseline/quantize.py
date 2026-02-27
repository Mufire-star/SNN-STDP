"""
8-bit Quantization for SNN MNIST Model
支持动态量化、静态量化、QAT三种方法
"""

import argparse
import struct
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 复用训练脚本中的配置和模型
import sys
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train as snn_train
from train import MNISTConvSNN, MNISTIdxDataset, TrainConfig
from spikingjelly.activation_based import encoding, functional


# ==================== 方法1: 动态量化 ====================
def dynamic_quantization(model: nn.Module, save_path: Path) -> nn.Module:
    """
    动态量化：最简单，权重量化，激活值实时量化
    优点：代码简单
    缺点：精度损失较大
    """
    print("=== Dynamic Quantization (INT8) ===")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.Conv2d, torch.nn.Linear},
        dtype=torch.qint8,  # 8-bit整数
    )
    
    # 保存
    torch.save(quantized_model.state_dict(), save_path)
    print(f"Saved dynamic quantized model: {save_path}")
    return quantized_model


# ==================== 方法2: 静态量化 (Post-Training Quantization) ====================
def static_quantization(
    model: nn.Module,
    calibration_loader: DataLoader,
    cfg: TrainConfig,
    save_path: Path
) -> nn.Module:
    """
    静态量化：需要calibration数据集来获取量化范围
    优点：精度好，推理快
    缺点：需要calibration数据
    """
    print("=== Static Quantization (INT8) ===")
    
    model.eval()
    
    # 1. 准备模型
    quantized_model = model.to('cpu')
    quantized_model.qconfig = torch.quantization.get_default_qconfig('x86')
    torch.quantization.prepare(quantized_model, inplace=True)
    
    # 2. Calibration - 收集激活值统计信息
    print("Calibration phase...")
    encoder = encoding.PoissonEncoder()
    with torch.no_grad():
        for idx, (x, y) in enumerate(calibration_loader):
            if idx % 10 == 0:
                print(f"  Calibrating batch {idx}/{len(calibration_loader)}")
            
            x = x.to(cfg.device)
            functional.reset_net(quantized_model)
            
            x_seq = torch.stack([encoder(x) for _ in range(cfg.T)], dim=0)
            x_seq = x_seq.cpu()  # 移到CPU用于量化
            _ = quantized_model(x_seq)
    
    # 3. 转换为量化模型
    torch.quantization.convert(quantized_model, inplace=True)
    
    # 保存
    torch.save(quantized_model.state_dict(), save_path)
    print(f"Saved static quantized model: {save_path}")
    return quantized_model


# ==================== 方法3: 量化感知训练 (QAT) ====================
def quantization_aware_training(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    num_qat_epochs: int = 5,
    save_path: Path = None,
) -> nn.Module:
    """
    QAT：在训练过程中模拟量化，精度最高
    优点：精度最优
    缺点：需要重新训练
    """
    print("=== Quantization Aware Training (INT8) ===")
    
    model = model.to(cfg.device)
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    
    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    encoder = encoding.PoissonEncoder()
    
    print(f"QAT with {num_qat_epochs} epochs...")
    for epoch in range(num_qat_epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            
            functional.reset_net(model)
            x_seq = torch.stack([encoder(x) for _ in range(cfg.T)], dim=0)
            y_seq = model(x_seq)
            fr = y_seq.mean(0)
            
            loss = torch.nn.functional.cross_entropy(fr, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            total_loss += float(loss.item())
        
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(cfg.device, non_blocking=True)
                y = y.to(cfg.device, non_blocking=True)
                
                functional.reset_net(model)
                x_seq = torch.stack([encoder(x) for _ in range(cfg.T)], dim=0)
                y_seq = model(x_seq)
                fr = y_seq.mean(0)
                pred = fr.argmax(1)
                
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        
        acc = correct / total
        print(f"  QAT Epoch {epoch + 1}/{num_qat_epochs} - Loss: {total_loss:.4f}, Acc: {acc:.4f}")
    
    # 转换为量化模型
    torch.quantization.convert(model, inplace=True)
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Saved QAT quantized model: {save_path}")
    
    return model


# ==================== 评估函数 ====================
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    cfg: TrainConfig,
    model_name: str = "Model"
) -> float:
    """评估量化模型的精度"""
    print(f"\n{model_name} Evaluation:")
    
    model.eval()
    encoder = encoding.PoissonEncoder()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            
            functional.reset_net(model)
            x_seq = torch.stack([encoder(x) for _ in range(cfg.T)], dim=0)
            y_seq = model(x_seq)
            fr = y_seq.mean(0)
            pred = fr.argmax(1)
            
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    
    acc = correct / total
    print(f"  Accuracy: {acc:.4f} ({correct}/{total})")
    return acc


# ==================== 模型大小对比 ====================
def get_model_size(model_path: Path) -> float:
    """获取模型文件大小(MB)"""
    if model_path.exists():
        return model_path.stat().st_size / (1024 * 1024)
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description='Quantize SNN MNIST model to INT8')
    parser.add_argument('--data-dir', type=str, default=str(Path(__file__).resolve().parent / 'datas' / 'MNIST' / 'raw'))
    parser.add_argument('--ckpt', type=str, default=str(Path(__file__).resolve().parent / 'weights' / 'fp32' / 'mnist_snn_baseline.pt'))
    parser.add_argument('--method', type=str, default='static', 
                       choices=['dynamic', 'static', 'qat'],
                       help='Quantization method')
    parser.add_argument('--save-dir', type=str, default=str(Path(__file__).resolve().parent / 'weights' / 'int8'))
    parser.add_argument('--qat-epochs', type=int, default=5, help='Number of QAT epochs')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载原模型
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device)
    ckpt_channels = int(ckpt.get('channels', 16))
    ckpt_T = int(ckpt.get('T', 8))

    model = MNISTConvSNN(channels=ckpt_channels, step_mode='m').to(args.device)
    model.load_state_dict(ckpt['state_dict'], strict=True)

    cfg = TrainConfig(
        T=ckpt_T,
        channels=ckpt_channels,
        batch_size=args.batch_size,
        epochs=1,
        lr=1e-3,
        device=args.device,
    )

    # 加载数据
    data_dir = Path(args.data_dir)
    test_ds = MNISTIdxDataset(
        data_dir / 't10k-images-idx3-ubyte',
        data_dir / 't10k-labels-idx1-ubyte',
    )

    # 对于 calibration / QAT，优先使用训练集；若缺失则回退到测试集
    train_images = data_dir / 'train-images-idx3-ubyte'
    train_labels = data_dir / 'train-labels-idx1-ubyte'
    has_train_set = train_images.exists() and train_labels.exists()
    if has_train_set:
        train_ds = MNISTIdxDataset(train_images, train_labels)
    else:
        train_ds = test_ds
        print('WARNING: train IDX files not found, fallback to test set for calibration/QAT.')

    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    
    # 只用前1000个样本用于 calibration / QAT，加速过程
    calib_ds = torch.utils.data.Subset(train_ds, range(min(1000, len(train_ds))))
    calib_loader = DataLoader(calib_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    # 记录原模型精度
    print("\n" + "="*60)
    original_acc = evaluate_model(model, test_loader, cfg, "Original FP32 Model")
    original_size = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    original_ckpt_size = get_model_size(Path(args.ckpt))

    # 执行量化
    print("\n" + "="*60)
    if args.method == 'dynamic':
        quantized_model = dynamic_quantization(model, save_dir / 'mnist_snn_q8_dynamic.pt')
    elif args.method == 'static':
        quantized_model = static_quantization(model, calib_loader, cfg, save_dir / 'mnist_snn_q8_static.pt')
    elif args.method == 'qat':
        quantized_model = quantization_aware_training(
            model, calib_loader, test_loader, cfg,
            num_qat_epochs=args.qat_epochs,
            save_path=save_dir / 'mnist_snn_q8_qat.pt'
        )

    # 评估量化模型
    print("\n" + "="*60)
    quantized_acc = evaluate_model(quantized_model, test_loader, cfg, f"Quantized ({args.method.upper()}) Model")
    quantized_size = get_model_size(save_dir / f'mnist_snn_q8_{args.method}.pt')

    # 输出对比结果
    print("\n" + "="*60)
    print("QUANTIZATION SUMMARY")
    print("="*60)
    print(f"Method:              {args.method.upper()}")
    print(f"Original Accuracy:   {original_acc:.4f}")
    print(f"Quantized Accuracy:  {quantized_acc:.4f}")
    print(f"Accuracy Drop:       {(original_acc - quantized_acc):.4f} ({(original_acc - quantized_acc)*100:.2f}%)")
    print(f"\nOriginal Model Size: {original_ckpt_size:.2f} MB")
    print(f"Quantized Model Size: {quantized_size:.2f} MB")
    print(f"Compression Ratio:   {original_ckpt_size/quantized_size:.2f}x")
    print(f"Size Reduction:      {(1 - quantized_size/original_ckpt_size)*100:.1f}%")
    print("="*60)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
