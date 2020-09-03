import numpy as np


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

    data_range = kwargs.pop("data_range", 1.0)

    mse = compute_mse(img1, img2)
    return 10.0 * np.log10(data_range * data_range / mse)
