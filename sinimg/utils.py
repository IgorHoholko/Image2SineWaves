import numpy as np
import cv2
from pathlib import Path
import yaml

CMAP = 'tab20c'


def map_intensity2sin_params(intensity, freq, A, freq_polinom_degree):
    intensity_normalized = intensity / 255 # to ~[0,1]
    intensity_normalized = intensity_normalized ** freq_polinom_degree  # make accent on lower intensity (high frequency)
    freq_normalized = (freq - freq.min()) / freq.max() # to ~[0,1]

    map_ = {}
    for i, intense_norm in enumerate(intensity_normalized):
        best_fit_idx = ((freq_normalized - intense_norm) ** 2).argmin()
        line = {i: {'freq': freq[best_fit_idx], "A": A[best_fit_idx]}}
        map_.update(line)
    return map_


def read_yaml(path):
    with open(path, 'r') as f:
        f_str = f.read()
        file = yaml.load(f_str, Loader=yaml.FullLoader)
    return file


def img_to_0_255_range(img):
    img = (img - img.min()) / img.max() # -> [0,1]
    img *= 255 # -> [0, 255]
    return np.array(img, dtype=np.uint8)


def save_history_imgs(history_imgs, out_dir):
    out_dir = Path(out_dir)
    for img_name, img in history_imgs.items():
        img = img_to_0_255_range(img)
        cv2.imwrite(str(out_dir / f'{img_name}.png'), cv2.applyColorMap(img, cv2.COLORMAP_BONE))