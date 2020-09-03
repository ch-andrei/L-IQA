import matlab_py.matlab_wrapper as mw

import os


def init_instance_fsim(**kwargs):
    matlab_eng = mw.get_matlab_instance()

    cwd = os.getcwd()  # assumes ".../iqa-tool/model" (or ".../{repo_name}/model") as runtime entry point
    matlab_eng.addpath(
        r'{}\iqa_metrics\fsim_matlab'.format(cwd),
        nargout=0)

    print("Initialized FSIM instance (MATLAB).")


def compute_fsim(img1, img2, **kwargs):
    data_range = kwargs.pop('data_range', 1.0)

    # FSIM assumes 0-255 dynamic range for the input images
    # rescale both images to 0-255 maintaining the original dynamic range difference ratio
    img1 *= 255.0 / data_range
    img2 *= 255.0 / data_range

    matlab_eng = mw.get_matlab_instance()
    out, err = mw.get_io_streams()

    path1, path2, tag = mw.imgs_to_unique_mat(img1, img2, extra_identifier='fsim')
    # Note: For grayscale images, the returned FSIM and FSIMc are the same
    fsim = matlab_eng.fsim_wrapper(path1, path2, tag, stdout=out, stderr=err)
    mw.remove_matlab_unique_mat(path1, path2)

    return fsim
