from DisplayModels.display_model_simul import DisplayDegradationModel, new_simul_params
from utils.image_processing.pu2_encoding import *

from utils.misc.timer import Timer
from utils.misc.logger import Logger

from iqa_metrics.metrics import *


_epsilon = 1e-6


class IqaTool(object):
    def __init__(self,
                 iqa_to_use=None,
                 display_model_device="",  # ex: 'samsung_tm800'
                 ddm_use_luminance=True,  # RGB or luminance
                 use_pu_encoding=True,
                 tmqi_use_original=False,  # use the original Matlab code or Python implementation for TMQI
                 lpips_use_gpu=True,
                 verbose=False
                 ):
        """
        :param iqa_to_use:
        :param display_model_device:
        :param ddm_use_luminance:
            normally, this should set to True, since PU encoding is intended to be used with luminance signal
        :param use_pu_encoding:
            normally, this should set to True, since PU encoding is required for accurate IQA in non-ideal conditions
        :param tmqi_use_original:
        :param lpips_use_gpu:
        :param fid_use_gpu:
        :param verbose:
        """

        self.log = Logger(verbose)

        # initialize display model
        self.ddm = DisplayDegradationModel(display_model_device=display_model_device)
        self.ddm_use_luminance = ddm_use_luminance
        self.use_pu_encoding = use_pu_encoding

        if iqa_to_use is not None:
            if isinstance(iqa_to_use, list) and len(iqa_to_use) > 0:
                self.iqa_to_use = iqa_to_use
            else:
                self.iqa_to_use = [iqa_to_use]
        else:
            # default metrics
            self.iqa_to_use = [
                iqa_mse,
                iqa_psnr,
                iqa_ssim,
                iqa_msssim,
                iqa_mssim_py,
                iqa_lpips,
                iqa_tmqi,
                iqa_fsim,
                iqa_vsi,
                iqa_mdsi,
                iqa_hdr_vdp
            ]

        iqa_initialize_metrics(self.iqa_to_use,
                               tmqi_use_original=tmqi_use_original,
                               lpips_use_gpu=lpips_use_gpu)

        # timers for computation runtime stats
        self.timers_iqa = {iqa_variant.name: Timer(iqa_variant.name) for iqa_variant in self.iqa_to_use}
        self.verbose = verbose

    def get_iqa_inputs(self, iqa_variant, img_L, dm, img_pu=None):
        """
        Pick Luminance or PU encoded inputs based on the queried IQA type and current configuration
        :param iqa_variant:
        :param img_L:
        :return:
        """
        # hdr-vdp requires luminance values in cd/m2, not PU encoded results
        if iqa_variant.name == iqa_hdr_vdp.name or not self.use_pu_encoding:
            data_range = dm.get_L_upper_bound()  # Note: HDR-VDP-2 does not use this
            self.log('Using DM luminance for iqa-variant {}'.format(iqa_variant.name))
            return img_L, data_range
        else:
            if img_pu is None:
                img_pu = pu2_encode_offset(img_L)
            data_range = pu2e_data_range
            self.log('Using PU-encoded DM luminance for iqa-variant {}'.format(iqa_variant))
            return img_pu, data_range

    def compute_iqa(self, img1, img2,
                    illuminant=None,
                    illumination_map=None,
                    apply_reflection=True,
                    apply_screen_dimming=False,
                    ddm_use_luminance=True,
                    ):
        """
        Use this function to compute IQA between two images.
        :param img1:
        :param img2:
        :param illuminant: ambient illumination conditions in lux
        :param illumination_map: illumination map if illumination is non-uniform (height x width, values in range 0-1)
        :param apply_reflection:
        :param apply_screen_dimming:
        :param ddm_use_luminance:
        :return:
        """
        sim_params = new_simul_params(
            illuminant=illuminant,
            illumination_map=illumination_map,
            use_luminance_only=ddm_use_luminance,
            apply_reflection=apply_reflection,
            apply_screen_dimming=apply_screen_dimming)

        return self.compute_iqa_custom(img1, img2, sim_params1=sim_params, sim_params2=sim_params)

    def compute_iqa_custom(self, img1, img2,
                           sim_params1=None, sim_params2=None,
                           dm1=None, dm2=None,
                           return_simulated_L=False,
                           iqa_to_use=None):
        """
        Use this function to compute IQA between two images given the custom simulation parameters.
        :param img1: reference image
        :param img2: test image
        :param sim_params1:
        :param sim_params2:
        :param dm1: display model to use for simulating image 1
        :param dm2: display model to use for simulating image 2
        :param return_simulated_L:
        :param iqa_to_use: which IQA metrics to compute
        :return:
        """

        if iqa_to_use is None:
            iqa_to_use = self.iqa_to_use

        img_1_L, dm1 = self.ddm.simulate_display(img1, sim_params1, dm=dm1, return_dm=True)
        img_2_L, dm2 = self.ddm.simulate_display(img2, sim_params2, dm=dm2, return_dm=True)

        output = {}
        for iqa_variant in iqa_to_use:
            iqa_name = iqa_variant.name
            iqa_compute_function = iqa_variant.compute_function

            timer_iqa = self.timers_iqa[iqa_name]
            timer_iqa.start()

            img1_iqa, data_range1 = self.get_iqa_inputs(iqa_variant, img_1_L, dm1)
            img2_iqa, data_range2 = self.get_iqa_inputs(iqa_variant, img_2_L, dm2)

            data_range = max(data_range1, data_range2)  # get the highest value between dm1 and dm2

            Q = iqa_func_wrapper(iqa_compute_function, img1_iqa, img2_iqa, data_range=data_range)

            self.log(
                "img1_iqa", img1_iqa.min(), img1_iqa.mean(), img1_iqa.max(), '\n',
                "img2_iqa", img2_iqa.min(), img2_iqa.mean(), img2_iqa.max(), '\n',
                'iqa value:', Q,
            )

            self.log("{} took {} seconds...".format(timer_iqa.name, timer_iqa.stop()))

            output[iqa_name] = Q

        return output if not return_simulated_L else (output, img_1_L, img_2_L)

    #########################################################################################################
    # THE FUNCTIONS BELOW ARE FOR SPEEDING UP IQA COMPUTATION FOR A CUSTOM DATASET OF IMAGES ################
    #########################################################################################################

    # this function computes IQA in bulk and is intended for the following two test scenarios (both are computed):
    # 1. reference image at reference conditions vs reference image at test conditions
    # 2. reference image at reference conditions vs test image at test conditions
    # this function is mainly used as optimization to speed up computation and is not as relevant for the IqaTool itself
    def _compute_iqa_bulk_pairs(self, img_ref_rgb, imgs_test_rgb, illuminations):
        """
        Computes bulk image quality assessment pairs (returns two values per condition) between two images for multiple
        conditions and multiple IQA metrics. IQA is computed for i) reference undegraded vs reference degraded, and ii)
        reference undegraded vs test degraded. Takes in a single reference RGB image, a single (or a list of -
        assumes one image per each illumination level) RGB test image(s), and a single (or a list of) illumination(s) to
        compute IQA for. If lists are provided for test images and illuminations, the two lists must have matching
        length. The function can also return simulated images with degradation as per illumination condition.
        :param img_ref_rgb: RGB unprocessed image
        :param imgs_test_rgb: list of or a single RGB processed image(s) for each illumination condition
        :param illuminations: list of or a single illumination condition(s) in lux
        :return:
        """

        # compute the undegraded reference image
        img_u_ref_L = self.ddm.display_model.simulate_display(
            data=img_ref_rgb,
            E_amb=0,  # irrelevant since inject_reflection=False
            illumination_map=None,
            illumination_map_weight_mode=None,
            use_luminance_only=self.ddm_use_luminance,
            inject_reflection=False,
            use_display_dimming=False)

        img_u_ref_pu = pu2_encode_offset(img_u_ref_L)

        # handle input types in list or single value format
        if type(illuminations) is not list:
            illuminations = [illuminations]

        # ensure that input formats match
        input_is_list = type(imgs_test_rgb) is list
        if input_is_list and len(imgs_test_rgb) != len(illuminations):
            if len(illuminations) == 1:
                illuminations = illuminations * len(imgs_test_rgb)  # match list size
            else:
                raise ValueError("Number of test images must match number of illuminations (1 image per condition).")
        elif not input_is_list:
            imgs_test_rgb = [imgs_test_rgb] * len(illuminations)

        # create dictionaries for storing outputs: maps illumination to results
        bulk_iqa_results = {}
        bulk_img_invs = {}

        # loop over test images and conditions
        for img_t_rgb, illumination in zip(imgs_test_rgb, illuminations):
            iqa_results, img_invs = self.__compute_iqa_bulk_pairs(img_ref_rgb, img_t_rgb, img_u_ref_pu, img_u_ref_L,
                                                                  illumination)
            bulk_iqa_results[illumination] = iqa_results
            bulk_img_invs[illumination] = img_invs

        return bulk_iqa_results, bulk_img_invs

    def __compute_iqa_bulk_pairs(self, img_r_rgb, img_t_rgb, img_u_ref_pu, img_u_ref_L, illumination):
        """
        Computes IQA function without recomputing non-degraded display simulation for img1 (ref), for speedup.
        When computing in bulk, non-degraded display simulation doesn't need to be recomputed for each illumination
        condition.
        :param img_r_rgb: unprocessed image
        :param img_t_rgb: processed image
        :param img_u_ref_pu:
        :param img_u_ref_L:
        :param illumination:
        :return:
        """

        # simulate distorted images at current ambient light conditions

        dm = self.ddm.display_model

        img_r_L = dm.simulate_display(
            data=img_r_rgb,
            E_amb=illumination,
            illumination_map=None,
            illumination_map_weight_mode=None,
            use_luminance_only=self.ddm_use_luminance,
            inject_reflection=True,
            use_display_dimming=True)

        img_t_L = dm.simulate_display(
            data=img_t_rgb,
            E_amb=illumination,
            illumination_map=None,
            illumination_map_weight_mode=None,
            use_luminance_only=self.ddm_use_luminance,
            inject_reflection=True,
            use_display_dimming=True)

        iqa_results = {}
        for iqa_variant in self.iqa_to_use:

            iqa_name = iqa_variant.name
            field1, field2 = ("u", "p")

            iqa_img_ref, _ = self.get_iqa_inputs(iqa_variant, img_u_ref_L, dm, img_pu=img_u_ref_pu)  # precomputed PU
            iqa_img_1, _ = self.get_iqa_inputs(iqa_variant, img_r_L, dm)
            iqa_img_2, data_range = self.get_iqa_inputs(iqa_variant, img_t_L, dm)

            timer_iqa = self.timers_iqa[iqa_name]
            timer_iqa.start()

            iqa_1 = iqa_func_wrapper(iqa_variant, iqa_img_ref, iqa_img_1, data_range=data_range)
            iqa_2 = iqa_func_wrapper(iqa_variant, iqa_img_ref, iqa_img_2, data_range=data_range)

            self.log(
                "illumination", illumination, "\n",
                "iqa_img_ref", iqa_img_ref.min(), iqa_img_ref.mean(), iqa_img_ref.max(), '\n',
                "iqa_img_1", iqa_img_1.min(), iqa_img_1.mean(), iqa_img_1.max(), '\n',
                "iqa_img_2", iqa_img_2.min(), iqa_img_2.mean(), iqa_img_2.max(), '\n',
                'iqa values', iqa_1, iqa_2
                )

            self.log("{} took {} seconds...".format(timer_iqa.name, timer_iqa.stop()))

            iqa_results[iqa_name] = {field1: iqa_1, field2: iqa_2}

        return iqa_results
