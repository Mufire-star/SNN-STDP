from pynq import Overlay, allocate
import numpy as np
from PIL import Image

# ========== 1. 加载 FPGA ==========
# 自动读取同目录的 snn_stdp.hwh
ol = Overlay("snn_stdp.bit")
ol.download()
print("FPGA 加载成功!")

# ========== 2. 读取单张图片 ==========
img = Image.open("2.jpg").convert("L").resize((28, 28))
test_image = np.array(img, dtype=np.uint8).reshape(-1)

# ========== 3. 推理 ==========
payload = np.empty(785, dtype=np.uint8)
payload[0] = 0
payload[1:] = test_image

in_buf = allocate(shape=(785,), dtype=np.uint8)
out_buf = allocate(shape=(10,), dtype=np.uint16)
in_buf[:] = payload
out_buf[:] = 0
in_buf.flush()
out_buf.flush()

dma = ol.axi_dma_0
dma.sendchannel.transfer(in_buf)
dma.recvchannel.transfer(out_buf)
dma.sendchannel.wait()
dma.recvchannel.wait()

out_buf.invalidate()
scores = np.array(out_buf, copy=True)
in_buf.freebuffer()
out_buf.freebuffer()

if np.all(scores == 0):
    raise RuntimeError(f"硬件返回全零得分，当前结果无效: {scores.tolist()}")

print(f"预测类别: {np.argmax(scores)}")
print(f"各类得分: {scores.tolist()}")
