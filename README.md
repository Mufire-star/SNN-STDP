# SNN-STDP
本项目当前已实现 **SNN 基线网络从软件训练到 FPGA bitstream 生成** 的完整流程（不含 STDP 学习规则）。

## 目前已实现的功能

### 1) 软件侧 SNN 训练/测试
- 路径：`python/SNN-baseline/`
- `train.py`：MNIST SNN 训练（默认 `epochs=10`，支持自动检查/下载 MNIST）
- `test.py`：模型测试
- 基线权重：`python/SNN-baseline/weights/fp32/mnist_snn_baseline.pt`

### 2) 8bit 量化流程
- 路径：`python/SNN-baseline/quantize.py`
- 支持 `static / dynamic / qat` 量化模式
- 默认量化输出目录：`python/SNN-baseline/weights/int8/`

### 3) HLS 版真 SNN IP（非 ANN 代理）
- 路径：`HW/baseline/snn_ip/`
- 核心文件：
	- `snn_top.cpp` / `snn_top.h`
	- `snn_top_tb.cpp`（C 仿真）
	- `weights/weights_generated.h`（由 checkpoint 导出）
- 权重导出脚本：`HW/baseline/export_snn_weights.py`

### 4) 已完成的优化与结果
- 完成 Conv2 定向优化（v2 方案）
- 生成脚本：`HW/baseline/snn_ip/build_snn_ip_v2_conv2.tcl`
- 参考结果：相对早期版本显著降低推理时延，且保持 C 仿真可用

### 5) Vivado 集成与 bitstream 产出
- Vivado 构建脚本：`HW/baseline/snn_ip/vivado/build_pynq_bit_v2.tcl`
- 标准输出目录：`HW/baseline/snn_ip/vivado/output_v2/`
	- `snn_v2.bit`
	- `snn_v2.hwh`

### 6) PYNQ 端调用示例
- 路径：`SW/pynq/`
- `single_image_benchmark.py`：单张灰度图推理 + 延迟计时（Jupyter 分块风格）

## 一键重建
- 脚本：`./rebuild_all.sh`
- 默认流程：数据检查 -> 训练 -> 量化 -> 权重导出 -> HLS -> Vivado
- 可用参数：`./rebuild_all.sh --help`

## 说明
- 当前仓库主线为 SNN 基线工程验证与硬件部署流程。
- 项目名含 STDP，但当前实现版本不包含 STDP 训练机制。
