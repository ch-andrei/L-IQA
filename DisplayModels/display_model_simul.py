from DisplayModels.display_model import DisplayModel, get_desired_L_max
# from utils.image_processing.pu2_encoding import *
from utils.image_processing.exposure_compensation import exposure_compensation
from utils.image_processing.color_spaces import srgb2lum

from utils.misc.timer import Timer
from utils.misc.logger import get_logger

from iqa_metrics.metrics import *

from collections import namedtuple

# assumed ideal illumination conditions in lux
DDM_IDEAL_ILLUMINANT = 400


# struct to hold simulation parameters
iqa_simul_params = namedtuple(
    'IqaSimulParams',
    [
        'illuminant',
        'display_max_luminance',
        'use_luminance_only',
        'apply_reflection',
        'apply_screen_dimming'
    ]
)


def new_simul_params(illuminant=None,
                     display_max_luminance=None,
                     use_luminance_only=False,
                     apply_reflection=True,
                     apply_screen_dimming=True
                     ):
    """
    :param illuminant: ambient conditions in lux (float or array)
    :param display_max_luminance: maximum luminance of the display (float or array); used for local dimming
    :param use_luminance_only: toggle for only using luminance channel
    :param apply_reflection: toggle for applying reflection (when False, Lrefl = 0)
    :param apply_screen_dimming:
        toggle for adaptive display luminance (display luminance depends on ambient illumination)
    :return:
    """
    if illuminant is None:
        illuminant = DDM_IDEAL_ILLUMINANT
    return iqa_simul_params(
        illuminant, display_max_luminance, use_luminance_only, apply_reflection, apply_screen_dimming
    )


class DisplayDegradationModel(object):
    def __init__(self,
                 display_model=None,
                 ):
        if display_model is None or display_model == "":
            self.display_model = DisplayModel()
        elif isinstance(display_model, str):
            self.display_model = DisplayModel(display_model)
        elif isinstance(display_model, DisplayModel):
            self.display_model = display_model
        else:
            raise TypeError(f"Unsupported DisplayModel configuration {display_model}.")

        self.timer_simul = Timer("Simulation")
        self.log = get_logger(name="DisplayDegradationModel")

    def get_sim_params(self, sim_params=None, dm=None):
        if sim_params is None:
            sim_params = new_simul_params(
                illuminant=None,
                use_luminance_only=False,
                apply_reflection=True,
                apply_screen_dimming=True,
            )

        if dm is None:
            dm = self.display_model

        return sim_params, dm

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

        sim_params, dm = self.get_sim_params(sim_params, dm)

        illuminant = sim_params.illuminant
        if (isinstance(illuminant, int) or isinstance(illuminant, float)) and illuminant < 0:
            # if negative illuminant value, assume input is already simulated. Do not simulate.
            self.log.info(f"Negative illuminant input ({illuminant}). "
                          f"Will not use DisplayModel simulation; assume input is already simulated.")
            img_L = img.copy()

        else:
            self.log.info(f"Simulating for condition: {illuminant} lux.")

            self.timer_simul.start()

            img_L = dm.display_simulation(
                V=img,
                E_amb=sim_params.illuminant,
                L_max=sim_params.display_max_luminance,
                use_luminance_only=sim_params.use_luminance_only,
                inject_reflection=sim_params.apply_reflection,
                use_display_dimming=sim_params.apply_screen_dimming
            )

            self.log.info("{} took {} seconds.".format(self.timer_simul.name, self.timer_simul.stop()))

        return (img_L, dm) if return_dm else img_L

    def simulate_displays_rgb(self, rgb_i, sim_params, dm=None, L_blk_inv=0, L_max_inv=1100):
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
        :param dm: display model for forward simulation
        :param dm_inv_L_range:
        :return:
        """

        if dm is None:
            dm = self.display_model

        # forward simulation
        img_L = self.simulate_display(rgb_i, sim_params, dm)

        # inverse simulation
        dm_inv = DisplayModel(L_contrast_ratio=-1, L_blk=L_blk_inv, L_max=L_max_inv)
        rgb_o = dm_inv.display_simulation_inverse(
            img_L,
            sim_params.illuminant,
            L_max=sim_params.display_max_luminance,
            use_luminance_only=sim_params.use_luminance_only,
            undo_reflection=False,
            undo_display_lum_profile=False,
        )

        return rgb_o
