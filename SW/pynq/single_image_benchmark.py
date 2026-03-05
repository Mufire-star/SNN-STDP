# 功能：
# 1) 加载 `snn_v2.bit/.hwh`
# 2) 读取 1 张灰度图并预处理为 28x28 uint8
# 3) 通过 `AXI DMA + DDR + snn_top` 完成 1 次推理
# 4) 输出分类分数、预测类别、单张耗时与 FPS

# %%
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pynq import Overlay, allocate

# 1) 配置路径
# - `BIT_PATH` 和 `HWH_PATH`：你的 bit/hwh 文件路径
# - `IMAGE_PATH`：要测试的灰度图路径（建议是 28x28 或可缩放到 28x28）

# %%
BIT_PATH = Path("/home/xilinx/snn/snn_v2.bit")
HWH_PATH = Path("/home/xilinx/snn/snn_v2.hwh")
IMAGE_PATH = Path("/home/xilinx/snn/test.png")

if not BIT_PATH.exists():
    raise FileNotFoundError(f"Bitstream not found: {BIT_PATH}")
if not HWH_PATH.exists():
    raise FileNotFoundError(f"HWH not found: {HWH_PATH}")
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")


# 2) DMA寄存器定义
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
DMA_SR_HALTED = 0x00000001
DMA_SR_ERR_MASK = 0x00007000


# 3) 工具函数

# %%
def preprocess_image(image_path: Path) -> np.ndarray:
    """Load and preprocess image to 28x28 uint8 array"""
    img = Image.open(image_path).convert("L").resize((28, 28), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    return arr.reshape(-1)  # Flatten to 784 bytes


def wait_dma_idle(dma, status_reg: int, timeout_s: float = 2.0) -> None:
    """Poll DMA status register until idle or error"""
    t0 = time.perf_counter()
    while True:
        sr = dma.read(status_reg)
        if sr & DMA_SR_ERR_MASK:
            raise RuntimeError(f"DMA error, status=0x{sr:08X}")
        if sr & DMA_SR_IDLE:
            return
        if (time.perf_counter() - t0) > timeout_s:
            raise TimeoutError(f"DMA timeout, status=0x{sr:08X}")
        time.sleep(0.001)


# ## 4) 加载 Overlay 并获取 DMA IP

ol = Overlay(str(BIT_PATH))

if "axi_dma_0" not in ol.ip_dict:
    raise KeyError("axi_dma_0 not found in overlay")

dma = ol.axi_dma_0
print(f"DMA base address: 0x{dma.mmio.base_addr:08X}")
print(f"DMA address range: 0x{dma.mmio.length:X}")

# ## 5) 分配DDR缓冲区（使用allocate从CMA分配连续物理内存）

# Input buffer: 784 bytes (28x28 image)
in_buf = allocate(shape=(784,), dtype=np.uint8)

# Output buffer: 20 bytes (10 classes * 2 bytes uint16)
out_buf = allocate(shape=(10,), dtype=np.uint16)

print(f"Input buffer physical address: 0x{in_buf.device_address:08X}")
print(f"Output buffer physical address: 0x{out_buf.device_address:08X}")

# ## 6) 单张推理 + 计时
# 流程：
# 1) 加载图像到输入缓冲区
# 2) 重置并启动DMA
# 3) 配置S2MM（接收端先配置）
# 4) 配置MM2S（发送端后配置，触发传输）
# 5) 等待完成
# 6) 读取输出分数

img_u8 = preprocess_image(IMAGE_PATH)
in_buf[:] = img_u8  # Copy image to DDR input buffer

# Reset DMA
dma.write(MM2S_DMACR, DMA_CR_RESET)
dma.write(S2MM_DMACR, DMA_CR_RESET)
time.sleep(0.01)

# Start DMA channels
dma.write(MM2S_DMACR, DMA_CR_RS)
dma.write(S2MM_DMACR, DMA_CR_RS)

# Start timing
start_t = time.perf_counter()

# Configure S2MM (receive) first
dma.write(S2MM_DA, out_buf.device_address)
dma.write(S2MM_LENGTH, 20)  # 10 classes * 2 bytes

# Configure MM2S (send) - this triggers the transfer
dma.write(MM2S_SA, in_buf.device_address)
dma.write(MM2S_LENGTH, 784)  # 28*28 bytes

# Wait for both channels to complete
wait_dma_idle(dma, MM2S_DMASR, timeout_s=2.0)
wait_dma_idle(dma, S2MM_DMASR, timeout_s=2.0)

elapsed_s = time.perf_counter() - start_t

# Read results from output buffer
scores = out_buf.copy()
pred = int(np.argmax(scores))

print("\n========== Results ==========")
print(f"Scores: {scores.tolist()}")
print(f"Predicted class: {pred}")
print(f"Latency: {elapsed_s * 1e3:.3f} ms")
print(f"FPS: {1.0 / elapsed_s:.2f}")
print("=============================\n")

# ## 7) 清理资源
in_buf.freebuffer()
out_buf.freebuffer()