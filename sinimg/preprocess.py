import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import defaultdict


def split_imgs2segments(img, n_clusters, n_iter):
    kmeans_model = KMeans(n_clusters=n_clusters, n_init=n_iter)
    flat_img = img.flatten()
    clusters = kmeans_model.fit_predict(flat_img[..., np.newaxis])

    img_segmented = clusters.reshape(img.shape)
    intensity = kmeans_model.cluster_centers_.flatten()


    # intensity_sorted_idxs = intensity.argsort()
    # intensity = intensity[intensity_sorted_idxs]
    # print(intensity)
    # print(intensity_sorted_idxs)
    #
    # image_sorted = np.zeros_like(img_segmented)
    # for i, class_new in enumerate(intensity_sorted_idxs):
    #     image_sorted[np.where(img_segmented == i)] = class_new

    return img_segmented, intensity


def fuse_classes(img, map_intens):
    # print(f"Number of classes BEFORE fusion: {len(np.unique(img))}")
    # freq_dict = defaultdict(list)
    # # find classes with same freq.
    # for class_, line in map_intens.items():
    #     freq_dict[line['freq']].append(class_)
    #
    # # fuse classes:
    # for classes_same in freq_dict.values():
    #     if len(classes_same) == 1:
    #         continue
    #     for class_ in classes_same[1:]:
    #         img[np.where(img == class_)] = classes_same[0]
    # print(f"Number of classes AFTER fusion: {len(np.unique(img))}")
    return img


def apply_morphology(img, kernel):
    # apply morphology openning operation for each segment starting from biggest to lowest
    vals, nums = np.unique(img,return_counts=True) #vals are sorted
    for val in vals:
        img_binary = np.zeros_like(img, dtype=np.uint8)
        img_binary[np.where(img == val)] = 1
        dilation = cv2.dilate(img_binary,kernel,iterations = 1)
        img[np.where(dilation)] = val
    return img



def devide_img2lines(img, height_line, block_width):
    h = height_line
    w = block_width

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
