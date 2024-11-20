import os
import numpy as np
from PIL import Image
from scipy.fftpack import idct

# 量化表
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


# 反量化函数
def dequantize(block, quantization_table):
    return block * quantization_table


# IDCT 2D 变换函数
def idct_2d(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


# 增采样函数
def chroma_upsampling_420(y, cb, cr):
    h, w = y.shape
    cb_upsampled = np.zeros((h, w))
    cr_upsampled = np.zeros((h, w))

    # 确保不会超出cb和cr的范围
    cb_h, cb_w = cb.shape
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            if i // 2 < cb_h and j // 2 < cb_w:
                cb_upsampled[i:i + 2, j:j + 2] = cb[int(i / 2), int(j / 2)]
                cr_upsampled[i:i + 2, j:j + 2] = cr[int(i / 2), int(j / 2)]

    return cb_upsampled, cr_upsampled


# 颜色空间转换函数 (YCbCr -> RGB)
def ycbcr_to_rgb(ycbcr):
    y, cb, cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
    r = y + 1.402 * (cr - 128)
    g = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return np.dstack((np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255))).astype(np.uint8)


# 解码函数
def decode_image(encoded_file_path, output_folder):
    # 读取编码文件
    data = np.load(encoded_file_path)
    y_quantized = data['y']
    cb_quantized = data['cb']
    cr_quantized = data['cr']

    # 反量化
    y_dequantized = [dequantize(block, luminance_quantization_table) for block in y_quantized]
    cb_dequantized = [dequantize(block, luminance_quantization_table[:4, :4]) for block in cb_quantized]
    cr_dequantized = [dequantize(block, luminance_quantization_table[:4, :4]) for block in cr_quantized]

    # IDCT变换
    y_idct = [idct_2d(block) for block in y_dequantized]
    cb_idct = [idct_2d(block) for block in cb_dequantized]
    cr_idct = [idct_2d(block) for block in cr_dequantized]

    # 重新组装图像
    h, w = y_idct[0].shape
    y_image = np.zeros((h * len(y_idct), w))
    for i, block in enumerate(y_idct):
        y_image[i * h:(i + 1) * h, :] = block

    # 增采样
    cb_upsampled, cr_upsampled = chroma_upsampling_420(y_image, cb_idct[0], cr_idct[0])

    # 颜色空间转换回RGB
    rgb_image = ycbcr_to_rgb(np.dstack((y_image, cb_upsampled, cr_upsampled)))

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 保存图像
    output_file_name = os.path.splitext(os.path.basename(encoded_file_path))[0] + '_decoded.jpg'
    output_path = os.path.join(output_folder, output_file_name)
    Image.fromarray(rgb_image).save(output_path)


# 主函数
def main():
    encoded_folder = "codepicture"  # 编码文件存放的文件夹路径
    output_folder = "decodepicture"  # 解码图片保存的文件夹路径

    for filename in os.listdir(encoded_folder):
        if filename.endswith('.npz'):
            full_path = os.path.join(encoded_folder, filename)
            decode_image(full_path, output_folder)


if __name__ == "__main__":
    main()