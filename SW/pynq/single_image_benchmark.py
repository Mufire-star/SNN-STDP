# 功能：
# 1) 加载 `snn_v2.bit/.hwh`
# 2) 读取 1 张灰度图并预处理为 28x28 uint8
# 3) 通过 `AXI DMA + BRAM + snn_top` 完成 1 次推理
# 4) 输出分类分数、预测类别、单张耗时与 FPS

# %%
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pynq import MMIO, Overlay

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


# 2) DMA 寄存器与缓冲区布局
# 这里采用 BRAM 作为 DMA 源/目的缓存：
# - `SRC_OFF`：输入图像起始地址（784 字节）
# - `DST_OFF`：输出分数起始地址（10 类 * 2 字节 = 20 字节）

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
DMA_SR_ERR_MASK = 0x00007000

SRC_OFF = 0x0000
DST_OFF = 0x4000


# 3) 工具函数
# - `preprocess_image`：图像转 28x28 uint8
# - `write_bytes_to_bram`：写入输入缓存
# - `read_u16_from_bram`：读取 10 类输出分数
# - `wait_idle`：轮询 DMA 完成状态

# %%
def preprocess_image(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("L").resize((28, 28), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    return arr


def write_bytes_to_bram(bram_mmio: MMIO, off: int, data_u8: np.ndarray) -> None:
    n = len(data_u8)
    i = 0
    while i < n:
        b0 = int(data_u8[i])
        b1 = int(data_u8[i + 1]) if i + 1 < n else 0
        b2 = int(data_u8[i + 2]) if i + 2 < n else 0
        b3 = int(data_u8[i + 3]) if i + 3 < n else 0
        word = (b3 << 24) | (b2 << 16) | (b1 << 8) | b0
        bram_mmio.write(off + i, word)
        i += 4


def read_u16_from_bram(bram_mmio: MMIO, off: int, count: int) -> np.ndarray:
    raw = np.zeros(count * 2, dtype=np.uint8)
    for i in range(0, count * 2, 4):
        word = bram_mmio.read(off + i)
        raw[i] = word & 0xFF
        if i + 1 < raw.size:
            raw[i + 1] = (word >> 8) & 0xFF
        if i + 2 < raw.size:
            raw[i + 2] = (word >> 16) & 0xFF
        if i + 3 < raw.size:
            raw[i + 3] = (word >> 24) & 0xFF
    return raw.view(np.uint16)


def wait_idle(dma_mmio: MMIO, status_reg: int, timeout_s: float = 1.0) -> None:
    t0 = time.perf_counter()
    while True:
        sr = dma_mmio.read(status_reg)
        if sr & DMA_SR_ERR_MASK:
            raise RuntimeError(f"DMA error, status=0x{sr:08X}")
        if sr & DMA_SR_IDLE:
            return
        if (time.perf_counter() - t0) > timeout_s:
            raise TimeoutError(f"DMA timeout, status=0x{sr:08X}")

# ## 4) 加载 Overlay 并获取 IP 地址
# 需要在 bit/hwh 中包含：
# - `axi_dma_0`
# - `axi_bram_ctrl_0`

ol = Overlay(str(BIT_PATH))

if "axi_dma_0" not in ol.ip_dict:
    raise KeyError("axi_dma_0 not found in overlay")
if "axi_bram_ctrl_0" not in ol.ip_dict:
    raise KeyError("axi_bram_ctrl_0 not found in overlay")

dma_phys = ol.ip_dict["axi_dma_0"]["phys_addr"]
dma_span = ol.ip_dict["axi_dma_0"]["addr_range"]
bram_phys = ol.ip_dict["axi_bram_ctrl_0"]["phys_addr"]
bram_span = ol.ip_dict["axi_bram_ctrl_0"]["addr_range"]

dma = MMIO(dma_phys, dma_span)
bram = MMIO(bram_phys, bram_span)

print(f"DMA  : 0x{dma_phys:08X} span=0x{dma_span:X}")
print(f"BRAM : 0x{bram_phys:08X} span=0x{bram_span:X}")

# ## 5) 单张推理 + 计时
# 流程：
# 1) 图像写入 BRAM 源地址
# 2) 启动 DMA（S2MM 先配置，MM2S 后配置）
# 3) 轮询 MM2S/S2MM 完成
# 4) 读取 10 类分数并计算预测标签

img_u8 = preprocess_image(IMAGE_PATH).reshape(-1)  # 784 bytes

write_bytes_to_bram(bram, SRC_OFF, img_u8)

dma.write(MM2S_DMACR, DMA_CR_RESET)
dma.write(S2MM_DMACR, DMA_CR_RESET)
dma.write(MM2S_DMACR, DMA_CR_RS)
dma.write(S2MM_DMACR, DMA_CR_RS)

start_t = time.perf_counter()

dma.write(S2MM_DA, bram_phys + DST_OFF)
dma.write(S2MM_LENGTH, 20)      # 10 classes * 2 bytes

dma.write(MM2S_SA, bram_phys + SRC_OFF)
dma.write(MM2S_LENGTH, 784)     # 28*28 bytes input

wait_idle(dma, MM2S_DMASR, timeout_s=1.0)
wait_idle(dma, S2MM_DMASR, timeout_s=1.0)

elapsed_s = time.perf_counter() - start_t

scores = read_u16_from_bram(bram, DST_OFF, 10)
pred = int(np.argmax(scores))

print("scores:", scores.tolist())
print("pred  :", pred)
print(f"latency: {elapsed_s * 1e3:.3f} ms")
print(f"fps    : {1.0 / elapsed_s:.2f}")