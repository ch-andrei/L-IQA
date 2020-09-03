from iqa_metrics.reference_iqa import *
from iqa_metrics.ssim_matlab.ssim_wrapper import *
from iqa_metrics.lpips.lpips_wrapper import *
from iqa_metrics.tmqi_matlab.tmqi_wrapper import *
from iqa_metrics.fsim_matlab.fsim_wrapper import *
from iqa_metrics.vsi_matlab.vsi_wrapper import *
from iqa_metrics.mdsi_matlab.mdsi_wrapper import *
from iqa_metrics.hdrvdp.hdrvdp_wrapper import *

from collections import namedtuple


# IQA VARIANT FORMAT

# tuple to hold the required IQA function parameters
iqa_metric_variant = namedtuple('iqa_metric_variant', ['name', 'compute_function', 'init_function', 'init_kwargs'])


def new_iqa_variant(name, compute_function, init_function=None, init_kwargs=None):
    """
    use this function to initialize new IQA metrics (returns a iqa_metric_variant)
    :param name:
    :param compute_function: function to compute quality metric value Q between img1 and img2
    :param init_function: initialization function (will be called once before any calls to compute function
    :param init_kwargs: custom initialization parameters
    :return: iqa_metric_variant namedtuple
    """
    return iqa_metric_variant(name,
                              compute_function,
                              init_function,
                              init_kwargs={} if init_kwargs is None else init_kwargs
                              )


# IQA VARIANT DECLARATIONS

# Declare your IQA metrics here along with required compute and initialization functions;
# extra parameters can be passed via init_kwargs (dictionary)
iqa_mse = new_iqa_variant('MSE', compute_mse)
iqa_psnr = new_iqa_variant('PSNR', compute_psnr)
iqa_ssim = new_iqa_variant('SSIM', compute_ssim, init_instance_ssim)
iqa_msssim = new_iqa_variant('MSSSIM', compute_msssim, init_instance_ssim)
iqa_mssim_py = new_iqa_variant('Mean-SSIM_sk', compute_mean_ssim_py, init_instance_ssim, {'use_matlab': False})
iqa_lpips = new_iqa_variant('LPIPS', compute_lpips, init_instance_lpips)
iqa_tmqi = new_iqa_variant('TMQI', compute_tmqi, init_instance_tmqi)
iqa_fsim = new_iqa_variant('FSIM', compute_fsim, init_instance_fsim)
iqa_vsi = new_iqa_variant('VSI', compute_vsi, init_instance_vsi)
iqa_mdsi = new_iqa_variant('MDSI', compute_mdsi, init_instance_mdsi)
iqa_hdr_vdp = new_iqa_variant('HDR-VDP', compute_hdr_vdp, init_instance_hdr_vdp, {'hdrvdp_version': "3.0.6"})


# RUNTIME IQA FUNCTIONS

def iqa_initialize_metrics(iqa_to_use,
                           **kwargs
                           ):
    """
    :param iqa_to_use: list, iqa_metric_variant to be used
    :param kwargs: dictionary or any required arguments
    :return:
    """
    for iqa_variant in iqa_to_use:
        if iqa_variant.init_function is not None:
            iqa_variant.init_function(**iqa_variant.init_kwargs, **kwargs)


def iqa_func_wrapper(iqa_compute_function, img1, img2, **kwargs):
    """
    :param iqa_compute_function: iqa compute function to be called
    :param img1:
    :param img2:
    :param kwargs: any additional arguments for IQA functions
    :return:
    """

    # make a copy to avoid any modifications to the original images
    img_iqa_1 = img1.copy()
    img_iqa_2 = img2.copy()

    return iqa_compute_function(img_iqa_1, img_iqa_2, **kwargs)
