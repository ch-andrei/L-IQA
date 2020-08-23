from iqa_metrics.reference_iqa import compute_mse, compute_psnr
from iqa_metrics.ssim_matlab.ssim_wrapper import *
from iqa_metrics.lpips_wrapper.lpips_wrapper import *
from iqa_metrics.tmqi_matlab.tmqi_wrapper import *
# from iqa_metrics.python_wrappers.hdrvdp2_wrapper import *
from iqa_metrics.fsim_matlab.fsim_wrapper import *
from iqa_metrics.vsi_matlab.vsi_wrapper import *
from iqa_metrics.mdsi_matlab.mdsi_wrapper import *

from collections import namedtuple

################## IQA VARIANT DECLARATIONS  ##################

# struct to hold standard iqa function information
iqa_function_variant = namedtuple('iqa_function_variant', ['name', 'suffix', 'function'])

iqa_mse = iqa_function_variant(name='MSE', suffix='', function=compute_mse)
iqa_psnr = iqa_function_variant(name='PSNR', suffix='', function=compute_psnr)
iqa_ssim = iqa_function_variant(name='SSIM', suffix='', function=compute_ssim)
iqa_msssim = iqa_function_variant(name='MSSSIM', suffix='', function=compute_msssim)
iqa_mssim_py = iqa_function_variant(name='Mean-SSIM_sk', suffix='', function=compute_mean_ssim_py)
iqa_lpips = iqa_function_variant(name='LPIPS', suffix='', function=compute_lpips)
iqa_tmqi = iqa_function_variant(name='TMQI', suffix='', function=compute_tmqi)
# iqa_hdr_vdp_2 = iqa_function_variant(name='HDR-VDP-2', suffix='', function=compute_hdr_vdp_2)
iqa_fsim = iqa_function_variant(name='FSIM', suffix='', function=compute_fsim)
iqa_vsi = iqa_function_variant(name='VSI', suffix='', function=compute_vsi)
iqa_mdsi = iqa_function_variant(name='MDSI', suffix='', function=compute_mdsi)


def iqa_get_all_available_metrics():
    return [
        iqa_mse,
        iqa_psnr,
        # iqa_ssim,  # requires MATLAB
        # iqa_msssim,  # requires MATLAB
        iqa_mssim_py,
        iqa_lpips,
        # iqa_tmqi,  # requires MATLAB
        # iqa_hdr_vdp_2,  # requires MATLAB
        # iqa_fsim,  # requires MATLAB
        # iqa_vsi,  # requires MATLAB
        # iqa_mdsi,  # requires MATLAB
    ]


################## IQA METRIC INITIALIZATION ##################


def iqa_initialize_metrics(iqa_to_use,
                           tmqi_use_original,
                           lpips_use_gpu
                           ):
    # initialize IQA metrics
    if iqa_ssim in iqa_to_use or \
            iqa_msssim in iqa_to_use:
        init_instance_ssim(use_python=False)
    if iqa_mssim_py in iqa_to_use:
        init_instance_ssim(use_python=True)
    if iqa_tmqi in iqa_to_use:
        init_instance_tmqi(use_original=tmqi_use_original)
    if iqa_lpips in iqa_to_use:
        init_instance_lpips(use_gpu=lpips_use_gpu)
    # if iqa_hdr_vdp_2 in iqa_to_use:
    #     init_instance_hdr_vdp2()
    if iqa_fsim in iqa_to_use:
        init_instance_fsim()
    if iqa_vsi in iqa_to_use:
        init_instance_vsi()
    if iqa_mdsi in iqa_to_use:
        init_instance_mdsi()


################## IQA FUNCTION WRAPPER ##################


def iqa_func_wrapper(iqa_function, img1, img2, data_range=1.0, params=None):
    # make a copy to avoid any modifications to the original images
    img_iqa_1 = img1.copy()
    img_iqa_2 = img2.copy()
    if iqa_function is compute_mse:
        return compute_mse(img_iqa_1, img_iqa_2)
    elif iqa_function is compute_psnr or \
            iqa_function is compute_ssim or \
            iqa_function is compute_msssim or \
            iqa_function is compute_mean_ssim_py or \
            iqa_function is compute_lpips or \
            iqa_function is compute_fsim or \
            iqa_function is compute_vsi or \
            iqa_function is compute_mdsi:
        return iqa_function(img_iqa_1, img_iqa_2, data_range)
    elif iqa_function is compute_tmqi:
        return compute_tmqi(img_iqa_1, img_iqa_2)
    # elif iqa_function is compute_hdr_vdp_2:
    #     return compute_hdr_vdp_2(img_iqa_1, img_iqa_2, display_params=params)
    else:
        raise TypeError("Unsupported IQA function was called")
