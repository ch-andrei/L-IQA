import matlab_py.matlab_wrapper as mw
import os
import numpy as np
from utils.image_processing.color_spaces import rgb2lum

from skimage.metrics import structural_similarity as ssim_py


def init_instance_ssim(**kwargs):
    use_matlab = kwargs.pop('use_matlab', True)
    if use_matlab:
        matlab_eng = mw.get_matlab_instance()

        cwd = os.getcwd()  # assumes ".../iqa-tool/model" (or ".../{repo_name}/model") as runtime entry point
        matlab_eng.addpath(
            r'{}\iqa_metrics\ssim_matlab'.format(cwd),
            nargout=0)

        print("Initialized SSIM/MS-SSIM instance (MATLAB).")
    else:
        print("Initialized SSIM Python instance (will use structural_similarity from skimage.metrics).")


def compute_mean_ssim_py(img1, img2, **kwargs):
    return __compute_similarity(img1, img2, multiscale=True, use_python=True, **kwargs)


def compute_msssim(img1, img2, **kwargs):
    return __compute_similarity(img1, img2, multiscale=True, **kwargs)


def compute_ssim(img1, img2, **kwargs):
    return __compute_similarity(img1, img2, multiscale=False, **kwargs)


def __compute_similarity(img1, img2, multiscale=True, use_python=False, **kwargs):
    """
        Uses either the original SSIM implementation or its multi-scale variant
    :param img1:
    :param img2:
    :param multiscale: toggle between single- and multi- scale SSIM implementations (SSIM vs MSSIM)
                        Note: this only applies when using MATLAB; only MSSIM is available for Python
    :param use_python:
    :param kwargs:
    :return:
    """

    data_range = kwargs.pop('data_range', 1.0)

    if use_python:
        mat_range = 1.0
        ref = (img1 * mat_range / data_range).astype(np.float)
        A = (img2 * mat_range / data_range).astype(np.float)

        return ssim_py(ref, A, multichannel=True, data_range=mat_range)
    else:
        mat_range = 255.0

        # MATLAB version requires luminance inputs
        if len(img1.shape) == 3:
            img1 = rgb2lum(img1)
            img2 = rgb2lum(img2)

        ref = (img1 * mat_range / data_range).astype(np.int)
        A = (img2 * mat_range / data_range).astype(np.int)

        matlab_eng = mw.get_matlab_instance()
        out, err = mw.get_io_streams()

        path1, path2, tag = mw.imgs_to_unique_mat(A, ref, extra_identifier='ssim')
        ssim = matlab_eng.SSIM_wrapper(path1, path2, tag, multiscale, stdout=out, stderr=err)
        mw.remove_matlab_unique_mat(path1, path2)

        return ssim
