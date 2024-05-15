import logging

from DisplayModels.display_model_simul import DisplayDegradationModel, iqa_simul_params
from DisplayModels.display_model import DisplayModel
from utils.image_processing.pu21_encoding import PUTransform, PU21_TYPE_BANDING_GLARE

from utils.misc.timer import Timer
from utils.misc.logger import get_logger

from iqa_metrics.metrics import *


class IqaTool(object):
    def __init__(self,
                 iqa_variants=None,
                 device_profile_name=None,  # ex: 'samsung_tm800'
                 use_pu_encoding=True,
                 verbose=False
                 ):
        """
        :param iqa_variants:
        :param device_profile_name:
        :param use_pu_encoding:
            normally, this should be set to True, since PU encoding is required for accurate IQA in non-ideal conditions
        :param verbose:
        """

        self.log = get_logger(name="IqaTool", level=logging.DEBUG if verbose else logging.WARNING, stdout=verbose)

        # initialize display model
        self.ddm = DisplayDegradationModel(device_profile_name)

        self.use_pu_encoding = use_pu_encoding
        self.pu_encoder = PUTransform(
            encoding_type=PU21_TYPE_BANDING_GLARE,
            normalize=False,
            normalize_range_srgb=False,  # always rescale to [0.0, 1.0] if normalizing
            L_min_0=False,  # minimum L value is 0.0cd/m2 not 0.005cd/m2
        )

        if iqa_variants is not None:
            if isinstance(iqa_variants, list) and len(iqa_variants) > 0:
                self.iqa_variants = iqa_variants
            else:
                self.iqa_variants = [iqa_variants]
        else:
            # default metrics
            self.iqa_variants = [
                iqa_mse,
                iqa_psnr,
                iqa_ssim,
                iqa_ssim_py,
                iqa_msssim,
                iqa_msssim_py,
                iqa_tmqi,
                iqa_fsim,
                iqa_vsi,
                iqa_mdsi,
                iqa_hdr_vdp,
                iqa_lpips,
                iqa_vtamiq
            ]

        iqa_initialize_metrics(self.iqa_variants)

        # timers for computation runtime stats
        self.timers_iqa = {iqa_variant.name: Timer(iqa_variant.name) for iqa_variant in self.iqa_variants}

    def compute_iqa_custom(self, img1, img2,
                           sim_params1: iqa_simul_params=None, sim_params2: iqa_simul_params=None,
                           dm1: DisplayModel=None, dm2: DisplayModel=None,
                           return_simulated_L=False, iqa_variants=None):
        """
        Use this function to compute IQA between two images given the custom simulation parameters.
        :param img1: reference image
        :param img2: test image
        :param sim_params1:
        :param sim_params2:
        :param dm1: display model to use for simulating image 1
        :param dm2: display model to use for simulating image 2
        :param return_simulated_L:
        :param iqa_variants: which IQA metrics to compute
        :return:
        """

        if iqa_variants is None:
            iqa_variants = self.iqa_variants

        img_1_L, dm1 = self.ddm.simulate_display(img1, sim_params1, dm=dm1, return_dm=True)
        img_2_L, dm2 = self.ddm.simulate_display(img2, sim_params2, dm=dm2, return_dm=True)

        # compute PU-encoded luminance if PU-encoding is enabled and required by any of the used metrics
        img_1_pu = None
        img_2_pu = None
        if self.use_pu_encoding and any([iqa_variant.use_pu_encoding for iqa_variant in iqa_variants]):
            img_1_pu = self.pu_encoder(img_1_L)
            img_2_pu = self.pu_encoder(img_2_L)

        self.log.info(
            "img_1_L {}, {}, {}\n".format(img_1_L.min(), img_1_L.mean(), img_1_L.max()) +
            "img_2_L {}, {}, {}\n".format(img_2_L.min(), img_2_L.mean(), img_2_L.max()) +
            ("" if img_1_pu is None else
                "img_1_pu {}, {}, {}\n".format(img_1_pu.min(), img_1_pu.mean(), img_1_pu.max())) +
            ("" if img_2_pu is None else
                "img_2_pu {}, {}, {}\n".format(img_2_pu.min(), img_2_pu.mean(), img_2_pu.max()))
        )

        output = {}
        for iqa_variant in iqa_variants:
            iqa_name = iqa_variant.name
            iqa_compute_function = iqa_variant.compute_function

            timer_crt = self.timers_iqa[iqa_name]
            timer_crt.start()

            # 1. check if PU encoding is globally enabled
            # 2. check if PU encoding is required by the IQA metric
            use_pu_encoding = self.use_pu_encoding and iqa_variant.use_pu_encoding

            if use_pu_encoding:
                self.log.info(f'Using PU-encoded display luminance for iqa-variant={iqa_variant.name}')
                data_format = DATA_FORMAT_PU
                data_range = (self.pu_encoder.P_min, self.pu_encoder.P_max)
                img_1_iqa = img_1_pu
                img_2_iqa = img_2_pu

            else:
                self.log.info(f'Using simulated display luminance for iqa-variant={iqa_variant.name}')
                data_format = DATA_FORMAT_LUM
                data_range = (self.pu_encoder.L_min, self.pu_encoder.L_max)
                img_1_iqa = img_1_L
                img_2_iqa = img_2_L

            Q = iqa_func_wrapper(iqa_compute_function, img_1_iqa, img_2_iqa,
                                 data_format=data_format, data_range=data_range)

            self.log.info(f"{iqa_variant.name} Q={Q} (computed in {timer_crt.stop()} sec)\n")

            output[iqa_name] = Q

        return output if not return_simulated_L else (output, img_1_L, img_2_L)
