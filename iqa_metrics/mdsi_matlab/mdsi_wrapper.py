import matlab_py.matlab_wrapper as mw
from utils.image_processing.image_tools import ensure3d

import os


def init_instance_mdsi(**kwargs):
    matlab_eng = mw.get_matlab_instance()

    cwd = os.getcwd()  # assumes ".../iqa-tool/model" (or ".../{repo_name}/model") as runtime entry point
    matlab_eng.addpath(
        r'{}\iqa_metrics\mdsi_matlab'.format(cwd),
        nargout=0)

    print("Initialized MDSI instance (MATLAB).")


def compute_mdsi(img1, img2, **kwargs):
    data_range = kwargs.pop("data_range", 1.0)
    use_grayscale = kwargs.pop("use_grayscale", False)

    # MDSI assumes 0-255 dynamic range for the input images
    # rescale both images to 0-255 maintaining the original dynamic range difference ratio
    img1 *= 255.0 / data_range
    img2 *= 255.0 / data_range

    if len(img1.shape) < 3:
        use_grayscale = True

    # MDSI takes 3-channel RGB images
    img1 = ensure3d(img1)
    img2 = ensure3d(img2)

    matlab_eng = mw.get_matlab_instance()
    out, err = mw.get_io_streams()

    path1, path2, tag = mw.imgs_to_unique_mat(img1, img2, extra_identifier='mdsi')
    mdsi = matlab_eng.MDSI_wrapper(path1, path2, tag, use_grayscale, stdout=out, stderr=err)
    mw.remove_matlab_unique_mat(path1, path2)

    return 1 - mdsi
