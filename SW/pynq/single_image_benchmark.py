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
from pynq import MMIO, Overlay, allocate

# 1) 配置路径
# - `BIT_PATH` 和 `HWH_PATH`：你的 bit/hwh 文件路径
# - `IMAGE_PATH`：要测试的灰度图路径（建议是 28x28 或可缩放到 28x28）

# %%
BIT_PATH = Path("./snn_v2.bit")
HWH_PATH = Path("./snn_v2.hwh")
IMAGE_PATH = Path("./3.jpg")

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

INPUT_BYTES = 28 * 28
OUTPUT_CLASSES = 10
OUTPUT_BYTES_16 = OUTPUT_CLASSES * 2
OUTPUT_BYTES_32 = OUTPUT_CLASSES * 4


# 3) 工具函数

# %%
def preprocess_image(image_path: Path) -> np.ndarray:
    """Load and preprocess image to 28x28 uint8 array"""
    img = Image.open(image_path).convert("L").resize((28, 28), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    return arr.reshape(-1)  # Flatten to 784 bytes


def try_start_accelerator(ol: Overlay) -> None:
    ip_keys = list(ol.ip_dict.keys())

    candidates = []
    for name in ip_keys:
        low = name.lower()
        if "dma" in low or "ps" in low or "bram" in low:
            continue
        candidates.append(name)

    if len(candidates) == 0:
        return

    for name in candidates:
        info = ol.ip_dict.get(name, {})
        phys = info.get("phys_addr")
        span = info.get("addr_range", 0x1000)
        if phys is None:
            continue
        try:
            ctrl_mmio = MMIO(phys, max(span, 0x1000))
            ctrl = ctrl_mmio.read(0x00)
            ctrl_mmio.write(0x00, ctrl | 0x81)
        except Exception:
            pass


def run_once(dma, in_buf, out_buf, img_u8: np.ndarray, s2mm_length: int) -> np.ndarray:
    in_buf[:] = img_u8
    in_buf.flush()
    out_buf[:] = 0
    out_buf.flush()

    dma.write(MM2S_DMACR, DMA_CR_RS)
    dma.write(S2MM_DMACR, DMA_CR_RS)

    dma.write(MM2S_DMASR, dma.read(MM2S_DMASR) & DMA_SR_IRQ_MASK)
    dma.write(S2MM_DMASR, dma.read(S2MM_DMASR) & DMA_SR_IRQ_MASK)

    dma.write(S2MM_DA, out_buf.device_address)
    dma.write(S2MM_LENGTH, s2mm_length)

    dma.write(MM2S_SA, in_buf.device_address)
    dma.write(MM2S_LENGTH, INPUT_BYTES)

    wait_dma_idle(dma, MM2S_DMASR, timeout_s=2.0)
    wait_dma_idle(dma, S2MM_DMASR, timeout_s=2.0)

    out_buf.invalidate()
    return out_buf.copy()


def init_dma_channels(dma) -> None:
    dma.write(MM2S_DMACR, DMA_CR_RESET)
    dma.write(S2MM_DMACR, DMA_CR_RESET)

    t0 = time.perf_counter()
    while True:
        mm2s_cr = dma.read(MM2S_DMACR)
        s2mm_cr = dma.read(S2MM_DMACR)
        if ((mm2s_cr & DMA_CR_RESET) == 0) and ((s2mm_cr & DMA_CR_RESET) == 0):
            break
        if (time.perf_counter() - t0) > 0.01:
            raise TimeoutError(
                f"DMA reset timeout, MM2S_CR=0x{mm2s_cr:08X}, S2MM_CR=0x{s2mm_cr:08X}"
            )

    dma.write(MM2S_DMASR, dma.read(MM2S_DMASR) & DMA_SR_IRQ_MASK)
    dma.write(S2MM_DMASR, dma.read(S2MM_DMASR) & DMA_SR_IRQ_MASK)
    dma.write(MM2S_DMACR, DMA_CR_RS)
    dma.write(S2MM_DMACR, DMA_CR_RS)


def wait_dma_idle(dma, status_reg: int, timeout_s: float = 2.0) -> None:
    """Poll DMA status register until idle or error"""
    t0 = time.perf_counter()
    while True:
        sr = dma.read(status_reg)
        if sr & DMA_SR_ERR_MASK:
            flags = [name for bit, name in DMA_SR_ERR_BITS.items() if sr & bit]
            raise RuntimeError(
                f"DMA error, status=0x{sr:08X}, flags={','.join(flags) if flags else 'unknown'}"
            )
        if sr & DMA_SR_IRQ_MASK:
            dma.write(status_reg, sr & DMA_SR_IRQ_MASK)
        if sr & DMA_SR_IDLE:
            return
        if (time.perf_counter() - t0) > timeout_s:
            raise TimeoutError(f"DMA timeout, status=0x{sr:08X}")


# ## 4) 加载 Overlay 并获取 DMA IP

ol = Overlay(str(BIT_PATH))

if "axi_dma_0" not in ol.ip_dict:
    raise KeyError("axi_dma_0 not found in overlay")

dma = ol.axi_dma_0

# ## 5) 分配DDR缓冲区（使用allocate从CMA分配连续物理内存）

# Input buffer: 784 bytes (28x28 image)
in_buf = allocate(shape=(INPUT_BYTES,), dtype=np.uint8)

# Output buffers for protocol probing
out_buf16 = allocate(shape=(OUTPUT_CLASSES,), dtype=np.uint16)
out_buf32 = allocate(shape=(OUTPUT_CLASSES,), dtype=np.uint32)


# ## 6) 单张推理 + 计时
# 流程：
# 1) 加载图像到输入缓冲区
# 2) 重置并启动DMA
# 3) 配置S2MM（接收端先配置）
# 4) 配置MM2S（发送端后配置，触发传输）
# 5) 等待完成
# 6) 读取输出分数

img_u8 = preprocess_image(IMAGE_PATH)
try_start_accelerator(ol)
init_dma_channels(dma)

# Start timing
start_t = time.perf_counter()

raw16 = run_once(dma, in_buf, out_buf16, img_u8, OUTPUT_BYTES_16)
scores = raw16.astype(np.uint16)
decode_mode = "16-bit"

elapsed_s = time.perf_counter() - start_t

if np.all(scores == 0):
    start_t = time.perf_counter()
    raw32 = run_once(dma, in_buf, out_buf32, img_u8, OUTPUT_BYTES_32)
    low16 = (raw32 & 0xFFFF).astype(np.uint16)
    high16 = ((raw32 >> 16) & 0xFFFF).astype(np.uint16)
    scores = high16 if int(np.sum(high16)) > int(np.sum(low16)) else low16
    decode_mode = "32-bit packed"
    elapsed_s = time.perf_counter() - start_t

if np.all(scores == 0):
    start_t = time.perf_counter()
    raw16_inv = run_once(dma, in_buf, out_buf16, (255 - img_u8).astype(np.uint8), OUTPUT_BYTES_16)
    scores = raw16_inv.astype(np.uint16)
    decode_mode = "16-bit (inverted image retry)"
    elapsed_s = time.perf_counter() - start_t

pred = int(np.argmax(scores))

print("\n========== Results ==========")
print(f"Scores: {scores.tolist()}")
print(f"Predicted class: {pred}")
print(f"Latency: {elapsed_s * 1e3:.3f} ms")
print(f"FPS: {1.0 / elapsed_s:.2f}")
print("=============================\n")

# ## 7) 清理资源
in_buf.freebuffer()
out_buf16.freebuffer()
out_buf32.freebuffer()
