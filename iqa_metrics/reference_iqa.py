import numpy as np

from utils.image_processing.color_spaces import rgb2gray_matlab
from iqa_metrics.common import kwargs_get_data_params


CONTRAST_MAX_MIN = 1
CONTRAST_MICHELSON = 2


def compute_mse(img1, img2, **kwargs):
    """
    returns Mean Squared Error between two images
    :param img1:
    :param img2:
    :return:
    """
    return np.mean((img1 - img2) * (img1 - img2))


def compute_psnr(img1, img2, **kwargs):
    """
    Computes Peak Signal-to-Noise Ratio between two images.
    :param img1: image 1
    :param img2: image 2
    :param data_range: the maximum possible pixel value of the images.
                        For 8-bit images, range is 255.0; 1.0 for float images
    :return: psnr value
    """

    data_range, _ = kwargs_get_data_params(**kwargs)
    dmin, dmax = data_range
    data_range = dmax - dmin

    mse = compute_mse(img1, img2)
    return 10.0 * np.log10(data_range * data_range / mse)


def compute_contrast_ratio(img1, img2, **kwargs):
    return __compute_contrast_ratio(img2, contrast_type=CONTRAST_MAX_MIN)


def compute_contrast_michelson(img1, img2, **kwargs):
    return __compute_contrast_ratio(img2, contrast_type=CONTRAST_MICHELSON)


def __compute_contrast_ratio(img, contrast_type=CONTRAST_MAX_MIN):
    """
    simply the contrast ratio of the Test (Distorted) image
    :param img1:
    :param img:
    :param kwargs:
    :return:
    """

    # USE_PIXEL_RATIO controls the number (N) of pixels to use to determine contrast ratio.
    # Will use the average of N smallest/largest pixel values.
    # when ratio = 0, will use 1 pixel (absolute min and max, prone to outliers).
    # when ratio > 0, will average over N smallest and largest pixel values to compute min and max pixel levels
    USE_PIXEL_RATIO = 0.01

    pixels = img
    # get luminance
    if 3 == len(img.shape):
        pixels = rgb2gray_matlab(pixels)

    # Sort the pixel values in ascending order
    sorted_pixels = np.sort(pixels.reshape(-1))

    # Compute the average of the top and bottom 5% of pixel values
    num_pixels = len(sorted_pixels)
    num_pixels_cr = max(1, int(num_pixels * USE_PIXEL_RATIO))
    bot = np.mean(sorted_pixels[:num_pixels_cr])  # average of N smallest values
    top = np.mean(sorted_pixels[num_pixels - num_pixels_cr:])  # average of N largest values

    if contrast_type == CONTRAST_MAX_MIN:
        cr = top / np.maximum(bot, 1e-9)

    elif contrast_type == CONTRAST_MICHELSON:
        cr = (top - bot) / (top + bot)

    else:
        raise ValueError(f"Unsupported contrast type [{contrast_type}].")

    # print(f'[type={contrast_type}] cr={cr}, top={top}, bot={bot}, num_pixels_cr={num_pixels_cr}, USE_PIXEL_RATIO={USE_PIXEL_RATIO}, shape={pixels.shape}')

    return cr