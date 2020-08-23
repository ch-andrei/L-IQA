import matlab_py.matlab_wrapper as mw
from utils.image_processing.color_spaces import rgb2lum
import os, matlab_py
import numpy as np

from skimage.metrics import structural_similarity as ssim_py


def init_instance_ssim(use_python=False):
    if not use_python:
        matlab_eng = mw.get_matlab_instance()

        cwd = os.getcwd()  # assumes ".../iqa-tool/model" (or ".../{repo_name}/model") as runtime entry point
        matlab_eng.addpath(
            r'{}\iqa_metrics\ssim_matlab'.format(cwd),
            nargout=0)

        print("Initialized Matlab instance (SSIM/MS-SSIM).")
    else:
        print("Initialized Python instance (SSIM/MS-SSIM).")


def compute_mean_ssim_py(img1, img2, data_range):
    return compute_similarity(img1, img2, data_range, multiscale=True, use_python=True)


def compute_msssim(img1, img2, data_range):
    return compute_similarity(img1, img2, data_range, multiscale=True)


def compute_ssim(img1, img2, data_range):
    return compute_similarity(img1, img2, data_range, multiscale=False)


def compute_similarity(img1, img2, data_range, multiscale=True, use_python=False):
    """
    Uses either the original SSIM implementation or its multi-scale variant
    :param img1:
    :param img2:
    :param data_range:
    :return: ssim value
    """

    if use_python:
        mat_range = 1.0
        ref = (img1 * mat_range / data_range).astype(np.float)
        A = (img2 * mat_range / data_range).astype(np.float)

        return ssim_py(ref, A, multichannel=True, data_range=mat_range)
    else:
        mat_range = 255.0
        ref = (img1 * mat_range / data_range).astype(np.int)
        A = (img2 * mat_range / data_range).astype(np.int)

        matlab_eng = mw.get_matlab_instance()
        out, err = mw.get_io_streams()

        path1, path2, tag = mw.imgs_to_unique_mat(A, ref, extra_identifier='ssim')
        ssim = matlab_eng.SSIM_wrapper(path1, path2, tag, multiscale, stdout=out, stderr=err)
        mw.remove_matlab_unique_mat(path1, path2)

        return ssim
