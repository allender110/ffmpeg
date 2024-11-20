import os
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import subprocess

# 定义量化表
luminance_quantization_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


# 颜色空间转换函数
def rgb_to_ycbcr(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.dstack((y, cb, cr)).astype(np.float32)


# 色度子采样函数
def chroma_subsampling_420(ycbcr):
    y, cb, cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
    h, w = y.shape

    y_subsampled = y

    cb_subsampled = cb[::2, ::2]
    cr_subsampled = cr[::2, ::2]

    return y_subsampled, cb_subsampled, cr_subsampled


# DCT 2D 变换函数
def dct_2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


# 图像分块并进行DCT变换
def block_dct(image, block_size=8):
    h, w = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i + block_size, j:j + block_size]
            if block.shape == (block_size, block_size):
                blocks.append(dct_2d(block))
    return np.array(blocks)


# 量化函数
def quantize(block, quantization_table):
    return np.round(block / quantization_table).astype(int)


# 编码函数
def encode_image(image_path, output_folder):
    # 读取图像
    img = Image.open(image_path)
    img_array = np.array(img.convert('RGB'))

    # RGB到YCbCr
    ycbcr_image = rgb_to_ycbcr(img_array)

    # 色度子采样
    y_subsampled, cb_subsampled, cr_subsampled = chroma_subsampling_420(ycbcr_image)

    # DCT变换
    y_blocks_dct = block_dct(y_subsampled)
    cb_blocks_dct = block_dct(cb_subsampled, block_size=4)
    cr_blocks_dct = block_dct(cr_subsampled, block_size=4)

    # 量化
    y_quantized = [quantize(block, luminance_quantization_table) for block in y_blocks_dct]

    # 这里省略了对Cb和Cr的量化，因为需要不同的量化表
    # 假设我们直接使用了Y的量化表来演示
    cb_quantized = [quantize(block, luminance_quantization_table[:4, :4]) for block in cb_blocks_dct]
    cr_quantized = [quantize(block, luminance_quantization_table[:4, :4]) for block in cr_blocks_dct]

    # 保存量化后的数据到一个临时文件
    np.savez(os.path.join(output_folder, os.path.basename(image_path) + '.npz'), y=y_quantized, cb=cb_quantized,
             cr=cr_quantized)


# 主函数
def main():
    source_folder = "picture"  # 源文件夹路径
    output_folder = "codepicture"  # 输出文件夹路径

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            full_path = os.path.join(source_folder, filename)
            encode_image(full_path, output_folder)


if __name__ == "__main__":
    main()