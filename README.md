# SNN-STDP

这个分支现在只保留一条主线：**面向 STDP 机制的 SNN HLS IP、Vivado bitstream 构建，以及 PYNQ 端 train-then-infer 验证**。

原来与监督训练、量化、旧版基线 bitstream 相关的文件已经移除。当前工程不再依赖外部 PyTorch checkpoint，硬件初始权重由 HLS 顶层内部确定性生成，更适合作为一个专注 STDP 机制验证的干净分支。

## 目录结构

- `HW/stdp_snn/`
  - `snn_top.cpp` / `snn_top.h`：STDP SNN 顶层 HLS 设计
  - `snn_top_tb.cpp`：基于 MNIST raw 文件的 C 仿真 testbench
  - `build_hls_stdp.tcl`：Vitis HLS 构建脚本
  - `vivado/build_bit_stdp.tcl`：Vivado bitstream 构建脚本
- `scripts/download_mnist.py`
  - 从公开镜像下载 MNIST raw 数据集
- `build_stdp.sh`
  - 一键执行数据下载、HLS 和 Vivado 构建
- `SW/pynq/stdp_overlay_demo.py`
  - PYNQ 端验证程序，会自动加载 overlay 并通过 AXI DMA 运行 `infer` / `train-then-infer`
- `SW/pynq/stdp_effect_benchmark.py`
  - PYNQ 端自动 benchmark，会比较 `infer-only`、`same-class train-then-infer` 和 `different-class train-then-infer`，自动判断 STDP 是否呈现正向作用

## 当前设计

- 网络结构：`Conv -> LIF -> Pool -> Conv -> LIF -> Pool -> FC -> LIF`
- 输入数据：MNIST 28x28 灰度图
- 编码方式：基于 LFSR 的泊松脉冲编码
- 默认综合档位：保守 profile，`C1 = 8`、`C2 = 8`、`T_STEPS = 4`
- 运行模式：
  - `MODE_INFER = 0`：直接推理
  - `MODE_TRAIN = 1`：先接收 `NUM_TRAIN_IMG` 张训练图做 STDP 更新，再接收 1 张测试图做推理
- 当前固定训练样本数：`NUM_TRAIN_IMG = 4`

## 快速开始

先下载 MNIST raw 数据：

```bash
python3 scripts/download_mnist.py --out data/mnist/raw
```

只验证 HLS：

```bash
./build_stdp.sh --skip-vivado
```

更保守地跑 HLS 和实现：

```bash
./build_stdp.sh --clock-ns 10 --jobs 1
```

完整生成 bitstream：

```bash
./build_stdp.sh
```

生成结果默认位于：

- `HW/stdp_snn/vivado/output_stdp/snn_stdp.bit`
- `HW/stdp_snn/vivado/output_stdp/snn_stdp.hwh`

## PYNQ 验证

在 PYNQ 板端运行：

```bash
python3 SW/pynq/stdp_overlay_demo.py
```

默认行为：

- 自动查找 `HW/stdp_snn/vivado/output_stdp/snn_stdp.bit`
- 自动查找 `data/mnist/raw`
- 依次执行 `infer-only` 和 `train-then-infer`
- 默认使用当前硬件常量要求的 `4` 张训练图
- 如果传入 `--save-dir`，会额外保存 `json + csv` 结果

常见用法：

```bash
python3 SW/pynq/stdp_overlay_demo.py --mode both
python3 SW/pynq/stdp_overlay_demo.py --mode train --train-split train --test-split test
python3 SW/pynq/stdp_overlay_demo.py --bit HW/stdp_snn/vivado/output_stdp/snn_stdp.bit --raw-dir data/mnist/raw
python3 SW/pynq/stdp_overlay_demo.py --save-dir ./logs --tag board_check
```

自动判断 STDP 机制是否“有作用”：

```bash
python3 SW/pynq/stdp_effect_benchmark.py
python3 SW/pynq/stdp_effect_benchmark.py --samples-per-class 3 --save-dir ./logs --tag stdp_eval
python3 SW/pynq/stdp_effect_benchmark.py --support-split train --test-split test --repeat 3
```

这个脚本会对每个测试样本自动做三次比较：

- `infer-only`
- `same-class support`：4 张同类训练图 + 1 张测试图
- `different-class support`：4 张异类训练图 + 1 张测试图

输出里会统计：

- `same_margin_gain_mean`
- `diff_margin_gain_mean`
- `same_minus_diff_margin_gain_mean`
- `same_better_than_diff_fraction`
- `verdict`

`verdict` 的含义：

- `evidence_for_stdp_effect`：同类支持集比异类支持集更常带来正向 margin 改善
- `no_clear_evidence`：有变化，但不够稳定或不够强
- `counter_evidence`：异类支持集并不比同类支持集更差，甚至更好

注意：

- 当前硬件里的权重只在一次 `MODE_TRAIN` 调用内部保留，下一次调用会重新初始化。
- 因此这个 benchmark 判断的是“单次 train-then-infer 会话里，STDP 是否对当前测试样本有正向帮助”，不是跨多次调用的长期在线学习效果。

最小上板步骤：

1. 把仓库或至少这几样文件放到板子上：
   `HW/stdp_snn/vivado/output_stdp/snn_stdp.bit`
   `HW/stdp_snn/vivado/output_stdp/snn_stdp.hwh`
   `data/mnist/raw/*`
   `SW/pynq/stdp_overlay_demo.py`
   `SW/pynq/stdp_effect_benchmark.py`
2. 确认板端 Python 能导入 `pynq` 和 `numpy`
3. 在工程根目录执行 `python3 SW/pynq/stdp_overlay_demo.py` 或 `python3 SW/pynq/stdp_effect_benchmark.py`

## 依赖

- `python3`
- `vitis-run` 或 `vitis_hls` 或 `vivado_hls`
- `vivado`

已按本机现有工具链整理为 `Vivado/Vitis 2025.2.1` 流程，目标器件仍是 `xczu7ev-ffvc1156-2-e`。

## 说明

- 这是一个 **STDP 机制验证工程**，不是监督训练精度工程。
- 由于 STDP 是无监督更新，`train-then-infer` 的结果主要用于观察硬件侧学习行为，不保证分类准确率稳定提升。
- 下载得到的 MNIST 数据和构建产物都默认不纳入 git 追踪。
