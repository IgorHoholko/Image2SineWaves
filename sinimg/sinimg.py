import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os
import shutil
import  functools

from .utils import  img_to_0_255_range, map_intensity2sin_params
from .preprocess import *


class SinImg:
    def __init__(self, config, save_intermediate_results):
        self.height_line = config['height_line']
        self.block_width = config['width_block_min']

        self.gk_size = config['gauss_kernel_size']
        self.mk_size = config['morphology_opening_kernel_size']

        self.n_clusters = config['n_clusters']
        self.n_iters = config['n_iters']

        self.freq_polinom_degree = config['freq_polinom_degree']
        self.freq_min = config['freq_min']
        self.freq_max = config['freq_max']
        self.min_amplitude = config['min_amplitude']

        self.save_intermediate_results =  save_intermediate_results

    def __call__(self, img):
        return self.process(img)


    def process(self, img):
        self.history_imgs = {}

        # apply Gauss Kernel
        kernel_gauss = np.ones((self.gk_size, self.gk_size), np.float32) / self.gk_size ** 2
        img_blured = cv2.filter2D(img, -1, kernel_gauss)

        # segment Image by color
        img_segmented, intensity = split_imgs2segments(img_blured.copy(), self.n_clusters, self.n_iters)
        print(len(np.unique(img_segmented)))

        # init map from intensity (color) to frequency
        freq = np.linspace(self.freq_min, self.freq_max, 150)
        A = np.linspace(self.height_line / 2, self.min_amplitude, 150)  # max amplitude = half of line
        map_intens = map_intensity2sin_params(intensity, freq, A, self.freq_polinom_degree)

        # PREPROCESS HERE
        # fuse classes with same frequency
        img_segmented_fused = fuse_classes(img_segmented.copy(), map_intens)

        # apply morphology opening operatio
        morphology_kernel = np.ones([self.mk_size, self.mk_size], dtype=np.uint8)
        img_processed1 = apply_morphology(img_segmented_fused.copy(), morphology_kernel)

        # split image to homogeneous lines which will be processed after to sine waves
        img_processed2 = devide_img2lines(img_processed1.copy(), self.height_line, self.block_width)

        if self.save_intermediate_results:
            self.history_imgs = {
                '1img_blured' : img_blured,
                '2img_segmented' : img_segmented,
                '3img_segmented_fused' :img_segmented_fused,
                '4img_morphology' : img_processed1,
                '5img_lined' : img_processed2
            }

        # compute ands save result image
        img_out = paralell_image_computation(img_processed2.copy(), map_intens, self.height_line, self.block_width)
        return img_out



def paralell_image_computation(img, map_intens, height_line, block_width):
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

    shutil.rmtree('_temp')
    return IMG



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

