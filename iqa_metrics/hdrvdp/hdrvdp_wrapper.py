import matlab_py.matlab_wrapper as mw

import os

from iqa_metrics.common import kwargs_get_data_params, DATA_FORMAT_LUM, ensure_single_channel_lum

HDR_VDP_VERSION_2_2_2 = '2.2.2'
HDR_VDP_VERSION_3_0_6 = '3.0.6'
HDR_VDP_VERSIONS_SUPPORTED = [HDR_VDP_VERSION_2_2_2, HDR_VDP_VERSION_3_0_6]


__hdr_vdp_version = None


def init_instance_hdr_vdp(**kwargs):
    global __hdr_vdp_version

    __hdr_vdp_version = kwargs.pop('hdrvdp_version', HDR_VDP_VERSION_3_0_6)

    if __hdr_vdp_version not in HDR_VDP_VERSIONS_SUPPORTED:
        raise (ValueError(f"HDR-VDP version [{__hdr_vdp_version}] is not supported. "
                          f"Supported versions: {HDR_VDP_VERSIONS_SUPPORTED}."))

    matlab_eng = mw.get_matlab_instance()

    cwd = os.getcwd()

    matlab_eng.addpath(
        r'{}\iqa_metrics\hdrvdp'.format(cwd),
        nargout=0)
    matlab_eng.addpath(
        r'{}\iqa_metrics\hdrvdp\{}'.format(cwd, __hdr_vdp_version),
        nargout=0)
    matlab_eng.addpath(
        r'{}\iqa_metrics\hdrvdp\{}\matlabPyrTools_1.4_fixed'.format(cwd, __hdr_vdp_version),
        nargout=0)

    if __hdr_vdp_version == HDR_VDP_VERSION_3_0_6:
        matlab_eng.addpath(
            r'{}\iqa_metrics\hdrvdp\{}\utils'.format(cwd, __hdr_vdp_version),
            nargout=0)
        matlab_eng.addpath(
            r'{}\iqa_metrics\hdrvdp\{}\data'.format(cwd, __hdr_vdp_version),
            nargout=0)

    print("Initialized Matlab instance (HDR-VDP {}).".format(__hdr_vdp_version))


def compute_hdr_vdp(img1, img2, **kwargs):
    global __hdr_vdp_version

    data_range, data_format = kwargs_get_data_params(**kwargs)

    if data_format != DATA_FORMAT_LUM:
        raise ValueError("HDR-VDP requires Luminance (cd/m2) input.")

    img1 = ensure_single_channel_lum(img1)
    img2 = ensure_single_channel_lum(img2)

    color_encoding = 'luminance'
    display_params = kwargs.pop('display_params', None)  # 4 value array or tuple, see below for more info
    # TODO: maybe refactor this such that s, w, h come from the display model?
    if display_params is None:
        s = 21  # screen size in inches
        w = 1920; h = 1080  # screen resolution
        d = 0.5  # distance to the display
    else:
        s, w, h, d = display_params

    matlab_eng = mw.get_matlab_instance()
    out, err = mw.get_io_streams()

    path1, path2, tag = mw.imgs_to_unique_mat(img1, img2, identifier='hdrvdp')
    res = matlab_eng.hdrvdp_wrapper(path1, path2, tag, __hdr_vdp_version, color_encoding, [s, w, h, d],
                                    stdout=out, stderr=err)  # redirect prints to dummy I/O streams
    mw.remove_matlab_unique_mat(path1, path2)

    return (res["Q"] / 100.0) if __hdr_vdp_version == HDR_VDP_VERSION_2_2_2 else res["Q"] / 10
