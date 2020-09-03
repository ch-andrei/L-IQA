from iqa_metrics.tmqi_matlab.TMQI import TMQI, TMQIr_m

import matlab_py.matlab_wrapper as mw
import os

from utils.image_processing.color_spaces import rgb2lum

# singleton instance to not reinitialize every time TMQI is called
tmqi_instance = None


def init_instance_tmqi(**kwargs):
    global tmqi_instance

    matlab_eng = mw.get_matlab_instance()

    use_original = kwargs.pop('use_original', True)

    if use_original:
        cwd = os.getcwd()  # assumes ".../iqa-tool/model" (or ".../{repo_name}/model") as runtime entry point
        matlab_eng.addpath(
            r'{}\iqa_metrics\tmqi_matlab'.format(cwd),
            nargout=0)
        print("Initialized TMQI instance (MATLAB).")
    else:
        if tmqi_instance is None:
            # choose between TMQI (original Python implementation by David Völgyes)
            # or TMQIr_m (modified by David Völgyes to fix inconsistent input scaling)
            tmqi_instance = TMQIr_m()
        print("Initialized TMQI instance (Python).")


def compute_tmqi(img1, img2, **kwargs):
    global tmqi_instance

    # no need to rescale inputs since TMQI claims to be dynamic range independent

    if tmqi_instance is None:
        # MATLAB version requires luminance inputs
        if len(img1.shape) == 3:
            img1 = rgb2lum(img1)
            img2 = rgb2lum(img2)

        # use Matlab version
        matlab_eng = mw.get_matlab_instance()
        out, err = mw.get_io_streams()

        path1, path2, tag = mw.imgs_to_unique_mat(img1, img2, extra_identifier='tmqi')
        Q = matlab_eng.tmqi_wrapper(path1, path2, tag, stdout=out, stderr=err)
        mw.remove_matlab_unique_mat(path1, path2)
    else:
        hdr = img1
        ldr = img2
        Q, S, N, s_local, s_maps = tmqi_instance(hdr, ldr)

    return Q
