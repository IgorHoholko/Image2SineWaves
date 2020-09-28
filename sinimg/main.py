import cv2
import numpy as np
import argparse
import shutil
import os
from pathlib import Path

from .utils import read_yaml, save_history_imgs
from .sinimg import SinImg

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

    parser.add_argument(
        "--save_intermediate_results", default=False, action="store_true", help="If true, do not use wandb for this run",
    )
    args = parser.parse_args()
    return args



def main():
    args = _parse_args()
    config = read_yaml(args.config)
    save_intermediate_results = args.save_intermediate_results

    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)


    img_orig = cv2.imread(args.image_path)
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

    img_processor = SinImg(config, save_intermediate_results)
    img_out = img_processor(img_gray)

    if save_intermediate_results:
        intermediate_results_dir = OUTPUT_DIR / 'intermediate_results'
        os.mkdir(intermediate_results_dir)
        save_history_imgs(img_processor.history_imgs, intermediate_results_dir)

    cv2.imwrite(str(OUTPUT_DIR / 'result_image.png'), img_out)

if __name__ == '__main__':
    main()
