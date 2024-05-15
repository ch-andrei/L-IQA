import matlab_py.matlab_wrapper as mw
from utils.image_processing.image_tools import ensure3d
from iqa_metrics.common import *

import os


def init_instance_vsi(**kwargs):
    matlab_eng = mw.get_matlab_instance()

    cwd = os.getcwd()
    matlab_eng.addpath(
        r'{}\iqa_metrics\vsi_matlab'.format(cwd),
        nargout=0)

    print("Initialized VSI instance (MATLAB).")


def compute_vsi(img1, img2, **kwargs):
    # Note: Matlab VSI code assumes input images contain values 0-255

    data_range, data_format = kwargs_get_data_params(**kwargs)
    img1 = transform_data_255(img1, data_range, data_format)
    img2 = transform_data_255(img2, data_range, data_format)

    # VSI takes 3-channel RGB images
    img1 = ensure3d(img1)
    img2 = ensure3d(img2)

    matlab_eng = mw.get_matlab_instance()
    out, err = mw.get_io_streams()

    path1, path2, tag = mw.imgs_to_unique_mat(img1, img2, identifier='vsi')
    vsi = matlab_eng.vsi_wrapper(path1, path2, tag, stdout=out, stderr=err)
    mw.remove_matlab_unique_mat(path1, path2)

    return vsi
