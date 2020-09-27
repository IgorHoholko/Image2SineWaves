import numpy as np
import cv2
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os
import shutil
import  functools
import yaml

CMAP = 'tab20c'


def read_yaml(path):
    with open(path, 'r') as f:
        f_str = f.read()
        file = yaml.load(f_str, Loader=yaml.FullLoader)
    return file



def map_intensity2sin_params(intensity, freq, A, freq_polinom_degree):
    intensity_normalized = (intensity - intensity.min()) / 255
    intensity_normalized = intensity_normalized ** freq_polinom_degree  # make accent on lower freq
    freq_normalized = (freq - freq.min()) / freq.max()

    map_ = {}
    for i, intense_norm in enumerate(intensity_normalized):
        best_fit_idx = ((freq_normalized - intense_norm) ** 2).argmin()
        line = {i: {'freq': freq[best_fit_idx], "A": A[best_fit_idx]}}
        map_.update(line)
    return map_



def paralell_image_computation(img, map_intens, height_line, block_width, output_dir ):
    h = height_line
    height_img = img.shape[0]
    img_lines = [img[i * h:(i + 1) * h] for i in range(height_img//h)]
    #     return img_lines
    if os.path.exists('_temp'):
        try:
            shutil.rmtree('_temp')
        except:
            pass

    os.mkdir('_temp')

    # for i in range(len(img_lines)):
    #     _sub_image_computation(
    #     img_lines[i], map_intens, output_name = f'temp/output_{i}')
    _ = Parallel(n_jobs=-1)(delayed(_sub_image_computation)(
        img_lines[i], map_intens, block_width, output_name=f'_temp/output_{i}') for i in range(len(img_lines)))

    imgs_paths = os.listdir('_temp')
    imgs_paths = sorted(imgs_paths, key=lambda path: int(path.split('_')[-1].split('.')[0]))
    imgs = [cv2.imread('_temp/' + path) for path in imgs_paths]
    IMG = functools.reduce(lambda a,b: np.vstack((a, b)), imgs)
    cv2.imwrite(str(output_dir / 'result_image.png'), IMG)

    shutil.rmtree('_temp')



def _sub_image_computation(img, map_intens, block_width, output_name='output'):
    plt.figure(figsize=np.array(img.shape)[::-1] / 1000, dpi=100)

    w = block_width
    width_img = img.shape[1]

    start = 0  # x axis
    prev_val = img[0, 0]
    e = 0
    for j in range(width_img // w):
        val = img[:, j * w:(j + 1) * w][0, 0]  # all nums are same inside of the block, take any
        freq, A = map_intens[val]['freq'], map_intens[val]['A']
        end = min(j * w, width_img)
        if val != prev_val:

            delta = end - start

            nf = round( (delta - e) / freq )
            delta_ = nf*freq

            if delta_ <= 0:
                e += start - end
                continue


            x = np.linspace(0, delta_, delta_)
            y = np.sin(2 * np.pi * x / freq) * A

            plt.plot(x+start, y, color='black', linewidth=.15)

            # plt.plot([start, start], [0, 6], c = 'r', linewidth=.1)
            # plt.plot([end+1, end+1], [0, 6], c='g', linewidth=.1)

            start += delta_
            prev_val = val
            e = delta_ - delta
        else:
            continue

    if end - start > 0:
        delta = end - start
        x = np.linspace(0, delta, delta * 4)
        y = np.sin(2 * np.pi * x / freq) * A
        plt.plot(x+start, y, color='black', linewidth=.15)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, .4)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')

    plt.savefig(f'{output_name}.png', bbox_inches='tight', pad_inches=0, dpi=1000)
    plt.cla()
    plt.clf()
    plt.close()

