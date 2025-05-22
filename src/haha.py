import numpy as np
import os
import cv2  # OpenCV 用于图像保存

# === 1. 加载 .npy 文件 ===
data = np.load("./dataset/Multi_regional_MeteoSat/MeteoSat_AF_2022.npy")  # shape: [N, T, C, H, W]
print("数据形状:", data.shape)

# === 2. 创建输出文件夹 ===
output_dir = "output_jpg"
os.makedirs(output_dir, exist_ok=True)

# === 3. 处理前几个样本（比如前1个） ===
for sample_idx in range(1):  # 改成更大的数导出多个样本
    sample = data[sample_idx]  # shape: [T, C, H, W]

    for t in range(sample.shape[0]):  # 遍历时间帧
        frame = sample[t]  # shape: [C, H, W]

        # === 4. 合成 RGB 图像 ===
        # 通常 C=3，frame[0]=B01，frame[1]=B02，frame[2]=B03
        img = np.transpose(frame, (1, 2, 0))  # [H, W, C]

        # === 5. 标准化到 0~255（防止图像过亮过暗） ===
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # 0~1
        img = (img * 255).astype(np.uint8)

        # === 6. 保存为 JPG ===
        save_path = os.path.join(output_dir, f"sample{sample_idx}_frame{t}.jpg")
        cv2.imwrite(save_path, img)

print("✅ 保存完成！")
