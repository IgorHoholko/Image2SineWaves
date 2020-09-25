import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import multiprocessing
import os
import shutil
import  functools


HEIGHT_LINE = 14
WIDTH_BLOCK_MIN = 5
N_CLUSTERS = 200

CMAP = 'tab20c'


def split_imgs2segments(img, n_clusters):
    kmeans_model = KMeans(n_clusters=n_clusters, n_init=1)
    flat_img = img.flatten()
    clusters = kmeans_model.fit_predict(flat_img[..., np.newaxis])

    img_segmented = clusters.reshape(img.shape)
    intensity = kmeans_model.cluster_centers_
    return img_segmented, intensity


def apply_morphology(img, kernel):
    # apply morphology openning operation for each segment starting from biggest to lowest
    vals, nums = np.unique(img,return_counts=True) #vals are sorted
    for val in vals:
        img_binary = np.zeros_like(img, dtype=np.uint8)
        img_binary[np.where(img == val)] = 1
        dilation = cv2.dilate(img_binary,kernel,iterations = 1)
        img[np.where(dilation)] = val
    return img


def devide_img2lines(img):
    h = HEIGHT_LINE
    w = WIDTH_BLOCK_MIN
    height_img = (img.shape[0] // h) * h
    width_img = (img.shape[1] // w) * w
    img_new = np.zeros([height_img, width_img], dtype=np.uint8)

    for i in range(height_img // h):
        for j in range(width_img // w):
            new_block = np.zeros([h, w], dtype=np.uint8)
            new_block.fill(
                np.unique(img[i * h:(i + 1) * h, j * w:(j + 1) * w])[0]
            )  # fill the block with the number prevailing in this block
            img_new[i * h:(i + 1) * h, j * w:(j + 1) * w] = new_block
    return img_new


def map_intensity2sin_params(intensity, freq, A):
    intensity_normalized = (intensity - intensity.min()) / 255
    intensity_normalized = intensity_normalized ** 3  # make accent on lower freq
    freq_normalized = (freq - freq.min()) / freq.max()

    map_ = {}
    for i, intense_norm in enumerate(intensity_normalized):
        best_fit_idx = ((freq_normalized - intense_norm) ** 2).argmin()
        line = {i: {'freq': freq[best_fit_idx], "A": A[best_fit_idx]}}
        map_.update(line)
    return map_


def _sub_image_computation(img, map_intens, quality=4, output_name='output'):
    plt.figure(figsize=np.array(img.shape)[::-1] / 1000, dpi=100)

    w = WIDTH_BLOCK_MIN
    width_img = img.shape[1]

    start = 0  # x axis
    prev_val = img[0, 0]

    for j in range(width_img // w):
        val = img[:, j * w:(j + 1) * w][0, 0]  # all nums are same inside of the block, take any
        freq, A = map_intens[val]['freq'], map_intens[val]['A']
        end = min(j * w, width_img)
        if val != prev_val:

            delta = end - start

            nf = math.floor( delta /freq )
            delta_ = nf*freq

            x = np.linspace(0, delta_, delta_ * quality)
            y = np.sin(2 * np.pi * x / freq) * A

            plt.plot(x+start, y, color='black', linewidth=.15)
            # plt.plot([start, start], [0, 6], c = 'r', linewidth=.1)

            start += delta_
            prev_val = val
        else:
            continue
    if end - start > 0:
        delta = end - start
        x = np.linspace(0, delta, delta * 4)
        y = np.sin(2 * np.pi * x / freq) * A
        plt.plot(x+start, y, color='black', linewidth=.15)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, .4)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.savefig(f'{output_name}.png', bbox_inches='tight', pad_inches=0, dpi=1000)
    plt.cla()
    plt.clf()
    plt.close()


def paralell_image_computation(img, map_intens, ):
    h = HEIGHT_LINE
    w = WIDTH_BLOCK_MIN
    height_img = img.shape[0]
    width_img = img.shape[1]
    img_lines = [img[i * h:(i + 1) * h] for i in range(height_img//h)]
    #     return img_lines
    if os.path.exists('temp'):
        try:
            shutil.rmtree('temp')
        except:
            pass

    os.mkdir('temp')

    # make_sin_img(
    # img_lines[0], map_intens, output_name = f'temp/output_{0}')
    _ = Parallel(n_jobs=5)(delayed(_sub_image_computation)(
        img_lines[i], map_intens, output_name=f'temp/output_{i}') for i in range(len(img_lines)))

    imgs_paths = os.listdir('temp')
    imgs_paths = sorted(imgs_paths, key=lambda path: int(path.split('_')[-1].split('.')[0]))
    imgs = [cv2.imread('temp/' + path) for path in imgs_paths]
    IMG = functools.reduce(lambda a,b: np.vstack((a, b)), imgs)
    cv2.imwrite('BIG_IMG.png', IMG)

    # shutil.rmtree('temp')

