from iqa_metrics.reference_iqa import *
from iqa_metrics.ssim_matlab.ssim_wrapper import *
from iqa_metrics.lpips.lpips_wrapper import *
from iqa_metrics.tmqi_matlab.tmqi_wrapper import *
from iqa_metrics.fsim_matlab.fsim_wrapper import *
from iqa_metrics.vsi_matlab.vsi_wrapper import *
from iqa_metrics.mdsi_matlab.mdsi_wrapper import *
from iqa_metrics.hdrvdp.hdrvdp_wrapper import *
from iqa_metrics.vtamiq.vtamiq_wrapper import *
from iqa_metrics.patch_sampling import get_default_sampler_config
from collections import namedtuple, OrderedDict




class IQAVariant(object):
    def __init__(
            self,
            name,
            compute_function,
            init_function=None,
            init_kwargs=None,
            use_pu_encoding=True
    ):
        self.name = name
        self.compute_function = compute_function
        self.init_function = init_function
        self.init_kwargs = {} if init_kwargs is None else init_kwargs
        self.use_pu_encoding = use_pu_encoding



# IQA VARIANT DECLARATIONS

# Declare your IQA metrics here along with required compute and initialization functions;
# extra parameters can be passed via init_kwargs (dictionary)


# IQA VARIANT DECLARATIONS

# Declare your IQA metrics here and asign the required compute and initialization functions
# extra parameters can be passed via init_kwargs (dictionary)
iqa_mse = IQAVariant(
    'MSE', compute_mse
)

iqa_mse_lum = IQAVariant(
    'MSE-LUM', compute_mse, use_pu_encoding=False  # disable PU encoding - use luminance
)

iqa_psnr = IQAVariant(
    'PSNR', compute_psnr
)

iqa_psnr_lum = IQAVariant(
    'PSNR-LUM', compute_psnr, use_pu_encoding=False  # disable PU encoding - use luminance
)

iqa_contrast_maxmin = IQAVariant(
    'ContrastMaxMin', compute_contrast_ratio, use_pu_encoding=False  # disable PU encoding - use luminance
)

iqa_contrast_michelson = IQAVariant(
    'ContrastMichelson', compute_contrast_michelson, use_pu_encoding=False  # disable PU encoding - use luminance
)

iqa_ssim = IQAVariant(
    'SSIM', compute_ssim, init_instance_ssim
)

iqa_ssim_py = IQAVariant(
    'SSIM-py', compute_ssim_py, init_instance_ssim,
    init_kwargs={'use_matlab': False}
)

iqa_msssim = IQAVariant(
    'MSSSIM', compute_msssim, init_instance_ssim
)

iqa_msssim_py = IQAVariant(
    'MSSSIM-py', compute_msssim_py, init_instance_ssim,
    init_kwargs={'use_matlab': False}
)

iqa_lpips = IQAVariant(
    'LPIPS', compute_lpips, init_instance_lpips,
    init_kwargs={
        "use_gpu": True,
    }
)

iqa_tmqi = IQAVariant(
    'TMQI', compute_tmqi, init_instance_tmqi,
    init_kwargs={
        "tmqi_use_original": True,  # use the original Matlab code or Python implementation for TMQI
    },
    use_pu_encoding=False  # disable PU encoding - use luminance
)

iqa_fsim = IQAVariant(
    'FSIM', compute_fsim, init_instance_fsim
)

iqa_vsi = IQAVariant(
    'VSI', compute_vsi, init_instance_vsi
)

iqa_mdsi = IQAVariant(
    'MDSI', compute_mdsi, init_instance_mdsi
)

iqa_hdr_vdp = IQAVariant(
    'HDR-VDP', compute_hdr_vdp, init_instance_hdr_vdp,
    init_kwargs={
        'hdrvdp_version': HDR_VDP_VERSION_3_0_6
    },
    use_pu_encoding=False  # disable PU encoding - use luminance
)

iqa_vtamiq = IQAVariant(
    IQA_VTAMIQ_NAME, compute_vtamiq, init_instance_vtamiq,
    init_kwargs={
        "use_gpu": True,
        "model_path": None,  # UPDATE THIS
        # "model_path": "./iqa_metrics/vtamiq/weights/vtamiq_pretrained.pth",  # UPDATE THIS
        "patch_count": 1024,
        "patch_num_scales": 5,  # controls how many scales are used for patch sampling
        "patch_sampler_config": get_default_sampler_config(),
        "parallel_runs": 16  # predict for N independent sets of patches (in parallel) and average the result
    }
)

# RUNTIME IQA FUNCTIONS


def iqa_initialize_metrics(iqa_variants, **kwargs):
    """
    :param iqa_variants: list, iqa_metric_variant to be used
    :param kwargs: dictionary or any required arguments
    :return:
    """
    for iqa_variant in iqa_variants:
        if iqa_variant.init_function is not None:
            iqa_variant.init_function(**iqa_variant.init_kwargs, **kwargs)


def iqa_func_wrapper(iqa_compute_function: callable, img1, img2, **kwargs):
    """
    :param iqa_compute_function: iqa compute function to be called
    :param img1:
    :param img2:
    :param data_range: tuple describing the input's dynamic range of form (min, max)
    :param data_format:
    :param kwargs: any additional arguments for IQA functions
    :return:
    """

    # make a copy to avoid any modifications to the original images
    img_iqa_1 = img1.copy()
    img_iqa_2 = img2.copy()

    return iqa_compute_function(img_iqa_1, img_iqa_2, **kwargs)
