import os
import cv2
import numpy as np
from scipy.fftpack import dct, idct


def rgb_to_ycbcr(img):
    """
    Convert an RGB image to YCbCr format.
    """
    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                  [-0.1687, -0.3313, 0.5],
                                  [0.5, -0.4187, -0.0813]])
    offset = np.array([0, 128, 128])
    ycbcr_img = np.dot(img, transform_matrix.T) + offset
    return np.clip(ycbcr_img, 0, 255).astype(np.uint8)


def subsample_420(ycbcr_img):
    """
    Perform 4:2:0 chroma subsampling.
    """
    y, cb, cr = cv2.split(ycbcr_img)
    cb = cv2.resize(cb, (cb.shape[1] // 2, cb.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    cr = cv2.resize(cr, (cr.shape[1] // 2, cr.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    return y, cb, cr


def blockwise_dct_and_quantize(channel, quant_matrix):
    """
    Perform block-wise DCT and quantization on a single channel.
    """
    h, w = channel.shape
    channel = channel.astype(np.float32) - 128
    quantized_blocks = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i:i + 8, j:j + 8]
            if block.shape != (8, 8):
                # Padding if not divisible by 8
                block = np.pad(block, ((0, 8 - block.shape[0]), (0, 8 - block.shape[1])), mode='constant')
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            quantized_block = np.round(dct_block / quant_matrix).astype(np.int16)
            quantized_blocks.append(quantized_block)
    return quantized_blocks


def save_encoded_image(encoded_data, output_path):
    """
    Save encoded blocks to a binary file.
    """
    with open(output_path, "wb") as f:
        for block in encoded_data:
            f.write(block.tobytes())


def jpeg_encode(input_dir, output_dir):
    """
    Perform JPEG encoding on all images in the input directory and save to output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    quant_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ])

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            # Step 1: Convert to YCbCr
            ycbcr_img = rgb_to_ycbcr(img)

            # Step 2: Subsample 4:2:0
            y, cb, cr = subsample_420(ycbcr_img)

            # Step 3: Blockwise DCT and Quantization
            y_blocks = blockwise_dct_and_quantize(y, quant_matrix)
            cb_blocks = blockwise_dct_and_quantize(cb, quant_matrix)
            cr_blocks = blockwise_dct_and_quantize(cr, quant_matrix)

            # Save encoded blocks
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.bin")
            save_encoded_image(y_blocks + cb_blocks + cr_blocks, output_path)

            print(f"Encoded and saved: {output_path}")


# 执行编码
input_dir = "picture"  # 提取关键帧的目录
output_dir = "codepicture"  # 编码后保存的目录

jpeg_encode(input_dir, output_dir)
