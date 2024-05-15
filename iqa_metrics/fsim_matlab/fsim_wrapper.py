import matlab_py.matlab_wrapper as mw
from iqa_metrics.common import *

import os


def init_instance_fsim(**kwargs):
    matlab_eng = mw.get_matlab_instance()

    cwd = os.getcwd()
    matlab_eng.addpath(
        r'{}\iqa_metrics\fsim_matlab'.format(cwd),
        nargout=0)

    print("Initialized FSIM instance (MATLAB).")


def compute_fsim(img1, img2, **kwargs):
    # Note: Matlab FSIM code assumes input images contain values 0-255

    data_range, data_format = kwargs_get_data_params(**kwargs)
    img1 = transform_data_255(img1, data_range, data_format)
    img2 = transform_data_255(img2, data_range, data_format)

    matlab_eng = mw.get_matlab_instance()
    out, err = mw.get_io_streams()

    path1, path2, tag = mw.imgs_to_unique_mat(img1, img2, identifier='fsim')
    # Note: For grayscale images, the returned FSIM and FSIMc are the same
    fsim = matlab_eng.fsim_wrapper(path1, path2, tag, stdout=out, stderr=err)
    mw.remove_matlab_unique_mat(path1, path2)

    return fsim
