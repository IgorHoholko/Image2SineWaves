from .utils import *
import cv2
import numpy as np


if __name__ == '__main__':

    img_orig = cv2.imread('imgs/1.jpg')
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    img = cv2.filter2D(img_gray, -1, np.ones((kernel_size, kernel_size), np.float32) / kernel_size ** 2)

    img_segmented, intensity = split_imgs2segments(img, N_CLUSTERS)
    img_processed1 = apply_morphology(img_segmented.copy(), np.ones([4,4], dtype=np.uint8))
    img_processed2 = devide_img2lines(img_processed1.copy())

    freq = np.linspace(4.5, 40, 100)
    A = np.linspace(HEIGHT_LINE / 2.5, 1, 100)
    map_intens = map_intensity2sin_params(intensity, freq, A)

    paralell_image_computation(img_processed2.copy(), map_intens)