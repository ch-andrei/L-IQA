from iqa_metrics.tmqi_matlab.TMQI import TMQIr_m
from iqa_metrics.common import normalize, kwargs_get_data_params, DATA_FORMAT_LUM
from utils.image_processing.color_spaces import rgb2lum

import matlab_py.matlab_wrapper as mw
import os


# singleton instance to not reinitialize every time TMQI is called
__tmqi_instance = None


def init_instance_tmqi(**kwargs):
    global __tmqi_instance

    matlab_eng = mw.get_matlab_instance()

    use_original = kwargs.pop('use_original', True)

    if use_original:
        cwd = os.getcwd()
        matlab_eng.addpath(
            r'{}\iqa_metrics\tmqi_matlab'.format(cwd),
            nargout=0)
        print("Initialized TMQI instance (MATLAB).")

    else:
        if __tmqi_instance is None:
            # choose between TMQI (original Python implementation by David Völgyes)
            # or TMQIr_m (modified by David Völgyes to fix inconsistent input scaling)
            __tmqi_instance = TMQIr_m()
        print("Initialized TMQI instance (Python).")


def compute_tmqi(img1, img2, **kwargs):
    data_range, data_format = kwargs_get_data_params(**kwargs)

    if data_format != DATA_FORMAT_LUM:
        raise ValueError("TMQI requires inputs in luminance (cd/m2) format.")

    # treat img1 as the ldr image and img2 as the hdr image
    img_ldr = 255 * normalize(img1, data_range) ** 1/2.2  # simple inverse display model for the "ldr" input
    img_hdr = img2  # no need to rescale the "hdr" input

    if __tmqi_instance is None:
        # MATLAB version requires luminance inputs
        if len(img_ldr.shape) == 3:
            img_ldr = rgb2lum(img_ldr)
            img_hdr = rgb2lum(img_hdr)

        # use Matlab version
        matlab_eng = mw.get_matlab_instance()
        out, err = mw.get_io_streams()

        path1, path2, tag = mw.imgs_to_unique_mat(img_hdr, img_ldr, identifier='tmqi')
        Q = matlab_eng.tmqi_wrapper(path1, path2, tag, stdout=out, stderr=err)
        mw.remove_matlab_unique_mat(path1, path2)

    else:
        Q, S, N, s_local, s_maps = __tmqi_instance(img_hdr, img_ldr)

    return Q
