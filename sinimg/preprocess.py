import numpy as np
import cv2
from sklearn.cluster import KMeans


def split_imgs2segments(img, n_clusters, n_iter):
    kmeans_model = KMeans(n_clusters=n_clusters, n_init=n_iter)
    flat_img = img.flatten()
    clusters = kmeans_model.fit_predict(flat_img[..., np.newaxis])

    img_segmented = clusters.reshape(img.shape)
    intensity = kmeans_model.cluster_centers_
    return img_segmented, intensity


def fuse_classes(img, ):
    pass


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
