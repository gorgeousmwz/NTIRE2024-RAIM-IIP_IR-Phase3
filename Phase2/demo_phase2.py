import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import cv2
import numpy as np
from arch.bpnkernel_arch import BPNKernel
from arch.fftformer_arch import fftformer

def rgb2gray(img):
    gray = np.mean(img, axis=2)
    # cv2.imwrite(r"/home/liujingyun/NTIRE/bpn/basicsr/results/BPN-1-1/a.jpg", 255*gray)
    # exit()
    return gray


def save_kernels_grid(blurry_image, kernels, image_name):
    """
     Draw and save CONVOLUTION kernels in the blurry image.
     Notice that computed kernels are CORRELATION kernels, therefore are flipped.
    :param blurry_image: Tensor (channels,M,N)
    :param kernels: Tensor (kernel_size,kernel_size,M,N)
    :param masks: Tensor (K,M,N)
    :return:
    """

    M, N, kernel_size, _ = np.shape(kernels)
    grid_to_draw = 0.4 * 1 + 0.6 * rgb2gray(blurry_image.transpose(1, 2, 0)).copy()
    grid_to_draw = np.repeat(grid_to_draw[None, :, :], 3, axis=0)
    for i in range(2 * kernel_size, M - 2 * kernel_size // 2, 2 * kernel_size):
        for j in range(2 * kernel_size, N - 2 * kernel_size // 2, 2 * kernel_size):

            kernel_ij = np.repeat(kernels[None, i, j, :, :], 3, axis=0)

            kernel_ij_norm = (kernel_ij - kernel_ij.min()) / (kernel_ij.max() - kernel_ij.min())

            grid_to_draw[
                0, i - kernel_size // 2 : i + kernel_size // 2 + 1, j - kernel_size // 2 : j + kernel_size // 2 + 1
            ] = (
                0.5 * kernel_ij_norm[0, ::-1, ::-1]
                + (1 - kernel_ij_norm[0, ::-1, ::-1])
                * grid_to_draw[
                    0, i - kernel_size // 2 : i + kernel_size // 2 + 1, j - kernel_size // 2 : j + kernel_size // 2 + 1
                ]
            )
            grid_to_draw[
                1:, i - kernel_size // 2 : i + kernel_size // 2 + 1, j - kernel_size // 2 : j + kernel_size // 2 + 1
            ] = (1 - kernel_ij_norm[1:, ::-1, ::-1]) * grid_to_draw[
                1:, i - kernel_size // 2 : i + kernel_size // 2 + 1, j - kernel_size // 2 : j + kernel_size // 2 + 1
            ]

    grid_to_draw = np.clip(grid_to_draw, 0, 1).squeeze()
    cv2.imwrite(image_name, cv2.cvtColor((255 * grid_to_draw.transpose(1, 2, 0)).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="datasets/phase2")
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument("--kernel_model_path", type=str, default="pretrained/net_kernel.pth")
    parser.add_argument("--restore_model_path", type=str, default="pretrained/net_g.pth")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    kernel_model = BPNKernel()
    restore_model = fftformer(dim=32)

    kernel_model.load_state_dict(torch.load(args.kernel_model_path)["params"], strict=True)
    restore_model.load_state_dict(torch.load(args.restore_model_path)["params"], strict=True)

    kernel_model = kernel_model.cuda()
    restore_model = restore_model.cuda()

    with torch.no_grad():
        for img_path in os.listdir(args.input_folder):
            img = cv2.imread(os.path.join(args.input_folder, img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img = img.cuda()
            kernel = kernel_model(img)
            deblur_img = restore_model(img, kernel)
            deblur_img = deblur_img.clamp(0, 1).squeeze().permute(1, 2, 0).cpu().numpy()
            deblur_img = cv2.cvtColor(deblur_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.output_folder, img_path), (deblur_img * 255).astype("uint8"))


if __name__ == "__main__":
    main()
