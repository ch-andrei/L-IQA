import matlab_py.matlab_wrapper as mw
import os

from scipy.signal import convolve2d
from iqa_metrics.ssim_matlab.ssim_py import structural_similarity as ssim_py
from iqa_metrics.common import *


def init_instance_ssim(**kwargs):
    use_matlab = kwargs.pop('use_matlab', True)
    if use_matlab:
        matlab_eng = mw.get_matlab_instance()

        cwd = os.getcwd()
        matlab_eng.addpath(
            r'{}\iqa_metrics\ssim_matlab'.format(cwd),
            nargout=0)

        print("Initialized contrast-based metric instance (MATLAB).")
    else:
        print("Initialized contrast-based Python instance.")


def compute_msssim_py(img1, img2, **kwargs):
    return __compute_similarity(img1, img2, multiscale=True, use_python=True, **kwargs)


def compute_msssim(img1, img2, **kwargs):
    return __compute_similarity(img1, img2, multiscale=True, **kwargs)


def compute_ssim(img1, img2, **kwargs):
    return __compute_similarity(img1, img2, multiscale=False, **kwargs)


def compute_ssim_py(img1, img2, **kwargs):
    return __compute_similarity(img1, img2, multiscale=False, use_python=True, **kwargs)


def __compute_similarity_py(ref, dist, multiscale=False):
    """
    :param ref: int image
    :param dist: int image
    :param multiscale: toggle between MS-SSIM or SSIM
    :return:
    """
    compute_ssim_py = lambda ref, dist: ssim_py(
        ref, dist, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255
    )

    if multiscale:
        # constants
        levels = 5
        weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

        msssim_array = np.zeros_like(weights)
        mcs_array = np.zeros_like(weights)

        downsample_filter = np.ones((2, 2), float) / 4.
        downsample = lambda img: convolve2d(img, downsample_filter, mode='same', boundary='symm')[::2, ::2]

        for i in range(levels):
            msssim, mcs = compute_ssim_py(ref, dist)
            msssim_array[i] = msssim
            mcs_array[i] = mcs
            ref = downsample(ref)
            dist = downsample(dist)

        return np.prod(mcs_array[:levels-1] ** weights[:levels-1]) * (msssim_array[-1] ** weights[-1])
    else:
        return compute_ssim_py(ref, dist)[0]


def __compute_similarity(img1, img2, multiscale=True, use_python=False, **kwargs):
    """
        Uses either the original SSIM implementation or its multi-scale variant.
        Note: all SSIM variants assume dynamic range is 255

    :param img1:
    :param img2:
    :param multiscale: toggle between single- and multi- scale SSIM implementations (SSIM vs MSSIM)
                        Note: this only applies when using MATLAB; only MSSIM is available for Python
    :param use_python:
    :param kwargs:
    :return:
    """

    data_range, data_format = kwargs_get_data_params(**kwargs)
    img1 = transform_data_255(img1, data_range, data_format)
    img2 = transform_data_255(img2, data_range, data_format)

    # convert to single channel
    img1 = ensure_single_channel_lum(img1)
    img2 = ensure_single_channel_lum(img2)

    if use_python:
        return __compute_similarity_py(img1, img2, multiscale)
    else:
        matlab_eng = mw.get_matlab_instance()
        out, err = mw.get_io_streams()

        path1, path2, tag = mw.imgs_to_unique_mat(img1, img2, identifier='ssim')
        ssim = matlab_eng.SSIM_wrapper(path1, path2, tag, multiscale, stdout=out, stderr=err)
        mw.remove_matlab_unique_mat(path1, path2)

        return ssim
