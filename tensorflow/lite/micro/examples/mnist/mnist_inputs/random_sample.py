import numpy as np
from PIL import Image

# 生成隨機的 28x28 灰階影像 (像素值 0~255)
random_image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)

# 轉換成 PIL Image
img = Image.fromarray(random_image, mode='L')

# 存成 sample.png
img.save("sample.png")

print("已經生成 sample.png")
