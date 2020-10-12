import matlab_py.matlab_wrapper as mw

import os


hdr_vdp_version = None


def init_instance_hdr_vdp(**kwargs):
    global hdr_vdp_version

    hdr_vdp_version = kwargs.pop('hdrvdp_version', '2.2.2')  # '3.0.6'
    hdr_vdp_supported_version = ['3.0.6', '2.2.2']

    if hdr_vdp_version not in hdr_vdp_supported_version:
        raise(ValueError("HDR-VDP version [{}] is unsupported; use {}.".format(hdr_vdp_version, hdr_vdp_supported_version)))

    matlab_eng = mw.get_matlab_instance()

    cwd = os.getcwd()  # assumes ".../iqa-tool/model" (or ".../{repo_name}/model") as runtime entry point

    matlab_eng.addpath(
        r'{}\iqa_metrics\hdrvdp'.format(cwd),
        nargout=0)
    matlab_eng.addpath(
        r'{}\iqa_metrics\hdrvdp\{}'.format(cwd, hdr_vdp_version),
        nargout=0)
    matlab_eng.addpath(
        r'{}\iqa_metrics\hdrvdp\{}\matlabPyrTools_1.4_fixed'.format(cwd, hdr_vdp_version),
        nargout=0)

    if hdr_vdp_version == '3.0.6':
        matlab_eng.addpath(
            r'{}\iqa_metrics\hdrvdp\{}\utils'.format(cwd, hdr_vdp_version),
            nargout=0)
        matlab_eng.addpath(
            r'{}\iqa_metrics\hdrvdp\{}\data'.format(cwd, hdr_vdp_version),
            nargout=0)

    print("Initialized Matlab instance (HDR-VDP, v{}).".format(hdr_vdp_version))


def compute_hdr_vdp(img1, img2, **kwargs):
    global hdr_vdp_version

    matlab_eng = mw.get_matlab_instance()
    out, err = mw.get_io_streams()

    display_params = kwargs.pop('display_params', None)  # 4 value array or tuple, see below for more info
    color_encoding = kwargs.pop('color_encoding', 'luminance')

    if color_encoding == 'luminance' and (len(img1.shape) > 2 or len(img2.shape) > 2):
        print("WARNING (HDR-VDP wrapper): using 'luminance' color_encoding requires single-channel Luminance (cd/m2) "
              "input but inputs with shapes {} and {} were provided. Will use 'sRGB-display' mode instead."
              .format(img1.shape, img2.shape))
        color_encoding = 'sRGB-display'

    # TODO: maybe refactor this such that s, w, h come from the display model?
    if display_params is None:
        s = 21  # screen size in inches
        w = 1920; h = 1080  # screen resolution
        d = 0.5  # distance to the display
    else:
        s, w, h, d = display_params

    path1, path2, tag = mw.imgs_to_unique_mat(img1, img2, extra_identifier='hdrvdp')
    res = matlab_eng.hdrvdp_wrapper(path1, path2, tag, hdr_vdp_version, color_encoding, [s, w, h, d],
                                    stdout=out, stderr=err)  # redirect prints to dummy I/O streams
    mw.remove_matlab_unique_mat(path1, path2)

    return (res["Q"] / 100.0) if hdr_vdp_version == '2.2.2' else res["Q"] / 10
