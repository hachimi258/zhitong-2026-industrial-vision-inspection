import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# ==================== 可调整的参数区域 ====================
PARAMS = {python
    # 基础设置
    "input_dir": "e:/desktop/demo/1",       # 原始图片目录（含子文件夹）
    "output_dir": "e:/desktop/demo/2",      # 增强后保存目录（自动创建相同子结构）
    "num_augmented_per_image": 5,           # 每张原图生成几张增强图
    
    # 几何增强
    "random_rotate_deg": 15,                # 随机旋转角度范围 (-deg, +deg)
    "random_crop_scale": (0.9, 1.1),        # 随机裁剪缩放范围 (最小比例, 最大比例)
    "random_horizontal_flip": False,        # 是否随机水平翻转
    "random_vertical_flip": False,          # 是否随机垂直翻转
    
    # 色彩增强
    "brightness_delta": 0.2,                # 亮度变化范围 [-delta, +delta]
    "contrast_delta": 0.1,                  # 对比度变化范围 [1-delta, 1+delta]
    "saturation_delta": 0.1,                # 饱和度变化范围 [1-delta, 1+delta]
    "hue_delta": 0.1,                       # 色调变化范围 [-delta, +delta] (单位：度)
    
    # 其他增强
    "random_noise_std": 5,                  # 高斯噪声标准差 (0表示不加)
    "random_blur_ksize": 3,                 # 高斯模糊核大小 (奇数，0表示不加)
}
# ========================================================

def imread_chinese(path):
    """支持中文路径的图片读取，返回 BGR 格式 numpy 数组"""
    img = Image.open(path)
    # PIL 读取为 RGB，转换为 OpenCV 的 BGR
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def imwrite_chinese(path, img_bgr):
    """支持中文路径的图片保存，img_bgr 为 BGR 格式 numpy 数组"""
    # 转换为 RGB 并用 PIL 保存
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(img_rgb).save(path)

def apply_random_rotate(image, max_deg):
    """随机旋转"""
    angle = np.random.uniform(-max_deg, max_deg)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated

def apply_random_crop_resize(image, scale_range):
    """随机裁剪并缩回原尺寸"""
    h, w = image.shape[:2]
    crop_scale = np.random.uniform(scale_range[0], scale_range[1])
    new_h, new_w = int(h * crop_scale), int(w * crop_scale)
    
    # 确保裁剪区域不超出边界
    top = np.random.randint(0, max(1, h - new_h + 1)) if h > new_h else 0
    left = np.random.randint(0, max(1, w - new_w + 1)) if w > new_w else 0
    cropped = image[top:top+new_h, left:left+new_w]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized

def apply_color_jitter(image, brightness, contrast, saturation, hue):
    """颜色抖动 (HSV 空间)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # 亮度 (V通道)
    brightness_factor = 1 + np.random.uniform(-brightness, brightness)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * brightness_factor, 0, 255)
    
    # 对比度 (通过线性拉伸)
    if contrast > 0:
        contrast_factor = 1 + np.random.uniform(-contrast, contrast)
        hsv[:,:,2] = 128 + (hsv[:,:,2] - 128) * contrast_factor
        hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    
    # 饱和度 (S通道)
    if saturation > 0:
        saturation_factor = 1 + np.random.uniform(-saturation, saturation)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation_factor, 0, 255)
    
    # 色调 (H通道)
    if hue > 0:
        hue_shift = np.random.uniform(-hue, hue)
        hsv[:,:,0] = (hsv[:,:,0] + hue_shift) % 180
    
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_gaussian_noise(image, std):
    """添加高斯噪声"""
    if std <= 0:
        return image
    noise = np.random.normal(0, std, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def apply_gaussian_blur(image, ksize):
    """高斯模糊"""
    if ksize <= 0 or ksize % 2 == 0:
        return image
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def augment_single_image(image):
    """对单张图片应用一系列随机增强"""
    aug_img = image.copy()
    
    # 几何增强
    if PARAMS["random_rotate_deg"] > 0:
        aug_img = apply_random_rotate(aug_img, PARAMS["random_rotate_deg"])
    if PARAMS["random_crop_scale"][0] < 1.0 or PARAMS["random_crop_scale"][1] > 1.0:
        aug_img = apply_random_crop_resize(aug_img, PARAMS["random_crop_scale"])
    if PARAMS["random_horizontal_flip"] and np.random.rand() > 0.5:
        aug_img = cv2.flip(aug_img, 1)
    if PARAMS["random_vertical_flip"] and np.random.rand() > 0.5:
        aug_img = cv2.flip(aug_img, 0)
    
    # 色彩增强
    if any(v > 0 for v in [PARAMS["brightness_delta"], PARAMS["contrast_delta"], 
                           PARAMS["saturation_delta"], PARAMS["hue_delta"]]):
        aug_img = apply_color_jitter(aug_img, 
                                     PARAMS["brightness_delta"],
                                     PARAMS["contrast_delta"],
                                     PARAMS["saturation_delta"],
                                     PARAMS["hue_delta"])
    
    # 噪声与模糊
    if PARAMS["random_noise_std"] > 0:
        aug_img = apply_gaussian_noise(aug_img, PARAMS["random_noise_std"])
    if PARAMS["random_blur_ksize"] > 0:
        aug_img = apply_gaussian_blur(aug_img, PARAMS["random_blur_ksize"])
    
    return aug_img

def process_folder(input_folder, output_folder):
    """递归处理一个文件夹下的所有图片"""
    # 创建对应的输出文件夹
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    for item in input_folder.iterdir():
        if item.is_dir():
            # 递归处理子文件夹
            process_folder(item, output_folder / item.name)
        elif item.suffix.lower() in img_extensions:
            # 处理图片文件
            try:
                # 读取原始图片（支持中文路径）
                image = imread_chinese(str(item))
                base_name = item.stem  # 不含扩展名的文件名
                
                for i in range(PARAMS["num_augmented_per_image"]):
                    aug_img = augment_single_image(image)
                    # 生成新文件名，保留原始文件名（含中文）
                    save_name = f"{base_name}_aug{i+1}{item.suffix}"
                    save_path = output_folder / save_name
                    # 保存图片（支持中文路径）
                    imwrite_chinese(str(save_path), aug_img)
                    print(f"已保存: {save_path}")
            except Exception as e:
                print(f"处理文件 {item} 时出错: {e}")

def main():
    input_root = Path(PARAMS["input_dir"])
    output_root = Path(PARAMS["output_dir"])
    
    if not input_root.exists():
        print(f"错误：输入目录 {input_root} 不存在！")
        return
    
    print(f"开始处理，输入目录: {input_root}")
    print(f"输出目录: {output_root}")
    print(f"每张图片生成 {PARAMS['num_augmented_per_image']} 张增强图")
    print("=" * 50)
    
    # 处理整个目录结构
    process_folder(input_root, output_root)
    
    print(f"\n所有增强图片已保存到 {output_root}")
    print("你可以修改脚本开头的 PARAMS 字典来调整增强强度。")

if __name__ == "__main__":
    main()