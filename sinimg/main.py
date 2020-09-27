import cv2
import numpy as np
import argparse
import shutil
import os
from pathlib import Path

from .utils import (read_yaml, map_intensity2sin_params, paralell_image_computation)
from .preprocess import *

OUTPUT_DIR = Path(__file__).parents[1].resolve() / "output"


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path", '-i',
        type=str,
        help='Path to image you want to process',
    )
    parser.add_argument(
        "config",
        type=str,
        help='YAML file with parameters for the algorithm',
    )
    args = parser.parse_args()
    return args



def main():
    args = _parse_args()
    config = read_yaml(args.config)
    img_path = args.image_path

    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.mkdir(OUTPUT_DIR)


    img_orig = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

    gk_size = config['gauss_kernel_size']
    kernel_gauss = np.ones((gk_size, gk_size), np.float32) / gk_size ** 2
    img = cv2.filter2D(img_gray, -1, kernel_gauss)

    n_clusters = config['n_clusters']
    n_iters = config['n_iters']
    img_segmented, intensity = split_imgs2segments(img, n_clusters, n_iters)

    mk_size = config['morphology_opening_kernel_size']
    morphology_kernel = np.ones([mk_size, mk_size], dtype=np.uint8)
    img_processed1 = apply_morphology(img_segmented.copy(), morphology_kernel)

    height_line, block_width = config['height_line'], config['width_block_min']
    img_processed2 = devide_img2lines(img_processed1.copy(), height_line, block_width)

    if config.get('save_intermediate_results', False):
        intermediate_results_dir = OUTPUT_DIR / 'intermediate_results'
        os.mkdir(intermediate_results_dir)

        cv2.imwrite(str( intermediate_results_dir / 'img_segmented.png'), img_segmented)
        cv2.imwrite(str(intermediate_results_dir / 'img_morphology.png'), img_processed1)
        cv2.imwrite(str(intermediate_results_dir / 'img_lined.png'),      img_processed2)

    freq_polinom_degree = config['freq_polinom_degree']
    freq_min = config['freq_min']
    freq_max = config['freq_max']
    min_amplitude = config['min_amplitude']

    freq = np.linspace(freq_min, freq_max, 100)
    A = np.linspace(height_line / 2, min_amplitude, 100) # max amplitude = half of line
    map_intens = map_intensity2sin_params(intensity, freq, A, freq_polinom_degree)

    paralell_image_computation(img_processed2.copy(), map_intens, height_line, block_width,  OUTPUT_DIR)



if __name__ == '__main__':
    main()
