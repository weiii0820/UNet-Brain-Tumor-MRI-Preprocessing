import os
import cv2

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

# === 各種影像處理函式 ===

def apply_gaussian(image, kernel_size=(5, 5), sigma=0):
    img = cv2.GaussianBlur(image, kernel_size, sigma)
    return img.astype(np.float32) / 255.0

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img = clahe.apply(image)
    return img.astype(np.float32) / 255.0

def apply_hist_equalization(image):
    import cv2
    # 若是 float → 轉成 uint8；若已是 uint8，照舊
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # 執行直方圖等化（只接受 8-bit 單通道影像）
    img = cv2.equalizeHist(image)

    # 再次轉回 [0, 1] 的 float32，供模型使用
    return img.astype(np.float32) / 255.0


def preprocess_image(image, method='original'):
    """
    給模型用的影像處理函式,image 為灰階 NumPy 陣列，回傳處理後的 NumPy 陣列。
    method:'original', 'gaussian', 'clahe', 'he'
    """
    if method == 'original':
        return image
    elif method == 'gaussian':
        return apply_gaussian(image)
    elif method == 'clahe':
        return apply_clahe(image)
    elif method == 'he':
        return apply_hist_equalization(image)
    else:
        raise ValueError(f"未知的預處理方法：{method}")

# === ✅ 模型端可直接使用：完整流程（image path -> tensor） ===

def load_images_and_masks(image_size, image_dir, mask_dir, method='he'):
    """
    載入影像與 mask，進行：
    - Resize
    - 灰階轉換
    - 正規化 [0,1]
    - 可選的預處理方法（CLAHE、HE、Gaussian）
    - mask 二值化
    """
    image_list = sorted(os.listdir(image_dir))
    mask_list = sorted(os.listdir(mask_dir))

    x, y = [], []

    for img_name, mask_name in tqdm(zip(image_list, mask_list), desc="讀取資料", total=len(image_list)):
        # 載入影像（灰階）並轉為 NumPy 陣列
        img = load_img(os.path.join(image_dir, img_name), color_mode='grayscale', target_size=image_size)
        img = img_to_array(img).squeeze()  # shape: (H, W)
        img = preprocess_image(img, method)  # 使用指定的預處理
        img = np.expand_dims(img, axis=-1)  # shape: (H, W, 1)
        x.append(img)

        # 載入 mask 並轉為二值（0 或 1）
        mask = load_img(os.path.join(mask_dir, mask_name), color_mode='grayscale', target_size=image_size)
        mask = img_to_array(mask) / 255.0
        mask = np.round(mask)  # 強制二值化
        y.append(mask)

    return np.array(x), np.array(y)