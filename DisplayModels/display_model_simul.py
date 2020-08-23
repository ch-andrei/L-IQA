from DisplayModels.display_model import DisplayModel
from utils.image_processing.pu2_encoding import *
from utils.image_processing.exposure_compensation import exposure_compensation
from utils.image_processing.color_spaces import srgb2lum

from utils.misc.timer import Timer
from utils.misc.logger import Logger

from iqa_metrics.metrics import *


# assumed ideal illumination conditions in lux
ddm_ideal_iluminant = 300


# struct to hold simulation parameters
iqa_simul_params = namedtuple('iqa_sim_params',
                              ['illuminant',
                               'illumination_map',
                               'illumination_map_weight_mode',
                               'use_luminance_only',
                               'apply_reflection',
                               'apply_screen_dimming',
                               ])


def new_simul_params(illuminant=None,
                     illumination_map=None,
                     illumination_map_weight_mode='mean',
                     use_luminance_only=True,
                     apply_reflection=True,
                     apply_screen_dimming=True
                     ):
    if illuminant is None:
        illuminant = ddm_ideal_iluminant
    return iqa_simul_params(illuminant, illumination_map, illumination_map_weight_mode,
                            use_luminance_only, apply_reflection, apply_screen_dimming)


class DisplayDegradationModel(object):
    def __init__(self,
                 display_model_device,
                 verbose=False
                 ):
        self.display_model = DisplayModel(display_model_device)
        self.timer_simul = Timer("Simulation")
        self.log = Logger(verbose)

    def simulate_display(self, img, sim_params=None, dm=None, return_dm=False):
        """
        Runs the Display and Degradation Model (DDM) simulation for the input image given custom simulation and display
        parameters. Returns the simulated physical luminance values (cd/m2).
        :param img:
        :param sim_params:
        :param dm: display model
        :param return_dm: also return display model object (desireable when dm=None and want to know which dm is used)
        :return:
        """

        if sim_params is None:
            sim_params = new_simul_params(illuminant=None,
                                          illumination_map=None,
                                          use_luminance_only=True,
                                          apply_reflection=True,
                                          apply_screen_dimming=True)

        if dm is None:
            dm = self.display_model

        self.log("Simulating for condition:", sim_params.illuminant, "lux.")

        self.timer_simul.start()

        img_L = dm.display_simulation(
            data=img,
            E_amb=sim_params.illuminant,
            illumination_map=sim_params.illumination_map,
            illumination_map_weight_mode=sim_params.illumination_map_weight_mode,
            use_luminance_only=sim_params.use_luminance_only,
            inject_reflection=sim_params.apply_reflection,
            use_display_dimming=sim_params.apply_screen_dimming)

        self.log("{} took {} seconds.".format(self.timer_simul.name, self.timer_simul.stop()))

        return (img_L, dm) if return_dm else img_L

    def simulate_displays_rgb(self, rgb_i, sim_params, dm=None):
        """
        Use this function to simulate the appearance of a degraded image (perceptually degraded given the ambient
        conditions).
        This function operates in two steps:
        i) compute forward DDM simulation including display dimming and reflection
        ii) compute inverse DDM simulation without undoing display dimming and reflections,
        The perceptual effect of display dimming and reflections is not removed on the inverse DDM pass and thus
        this is visualized in the output rgb values.
        :param rgb_i: img rgb values
        :param sim_params:
        :param dm: display model
        :return:
        """

        if dm is None:
            dm = self.display_model

        img_L = self.simulate_display(rgb_i, sim_params, dm)

        rgb_o = dm.display_simulation_inverse(
            img_L,
            sim_params.illuminant,
            sim_params.illumination_map,
            sim_params.illumination_map_weight_mode,
            use_luminance_only=sim_params.use_luminance_only,
            undo_reflection=False,
            undo_display_lum_profile=False)

        return rgb_o
