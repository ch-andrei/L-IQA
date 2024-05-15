import numpy as np
import json

from utils.image_processing.image_tools import ensure3d
from utils.image_processing.color_spaces import rgb2lum, rgb2gray_matlab


DM_LUT_AMBIENT_LUX = None
DM_LUT_LUM_TARGET = None

DM_PROFILES_PATH = 'DisplayModels/profiles/'  # path to .json profiles folder
DM_AMBIENT_PROFILE = 'ambient_profile.json'  # LUT matching ambient condition to desired max screen luminance


def init_dm_luts():
    global DM_LUT_LUM_TARGET, DM_LUT_AMBIENT_LUX

    __lut_ambient_lux_field = "lut_ambient_lux"
    __lut_target_luminance_field = "lut_target_luminance"

    with open(DM_PROFILES_PATH + DM_AMBIENT_PROFILE, 'r') as ambient_profile_file:
        # load Lookup tables for display adaptive brightness profile
        __ambient_profile = json.load(ambient_profile_file)

        DM_LUT_AMBIENT_LUX = np.array(__ambient_profile[__lut_ambient_lux_field])
        DM_LUT_LUM_TARGET = np.array(__ambient_profile[__lut_target_luminance_field])


def get_desired_L_max(E_amb):
    if DM_LUT_LUM_TARGET is None or DM_LUT_LUM_TARGET is None:
        init_dm_luts()

    # determine Lmax from the amount of ambient illumination:
    # use the average level of illumination (if illumination is an array)
    E_mean = np.mean(E_amb) if isinstance(E_amb, np.ndarray) else E_amb

    # theoretical desired L_max value
    L_max = np.interp(E_mean, DM_LUT_AMBIENT_LUX, DM_LUT_LUM_TARGET)

    return L_max


def get_E_refl(E_amb, use_luminance_only):
    if isinstance(E_amb, np.ndarray) and len(E_amb.shape) == 3 and use_luminance_only:
        return rgb2lum(E_amb)  # convert reflection map to luminance if needed
    return E_amb


class DisplayModel(object):
    """
    This model simulates the visual stimuli emitted by a display given the displayed content, the display parameters,
    and the ambient illumination conditions.

    The simulation follows the approach described in Section 2.3 of "High Dynamic Range Imaging" (Mantiuk, Myszkowski,
    Seidel 2016) and implements the gamma-offset-gain (GOG) equation for display response:

    L = (Lmax - Lblack) * V ^ gamma + Lblack + Lrefl

        L is in units of cd/m2,
        V is the input luma signal in the range 0-1,
        Lmax is the maximum display luminance,
        Lblack is the display's black level luminance,
        Lrefl is the luminance resulting from ambient illumination computed as

    Lrefl = k * E_amb / PI

        k is the reflectivity constant,
        E_amb is the ambient illumination level in lux,
        PI is the mathematical constant 3.14159...
    """

    # constant namefields to fetch from .json profile
    __L_max_field = "L_max"
    __L_min_field = "L_min"
    __L_blk_field = "L_blk"
    __L_contrast_ratio_field = "L_contrast_ratio"
    __reflectivity_field = "k_reflectivity"
    __gamma_field = "gamma"

    # default device profile:
    # values used for "aosp_on_bullhead" (Nexus)
    __device_profile_name_d = "aosp_on_bullhead"
    __L_max_d = 460
    __L_min_d = 1.811
    __reflectivity_d = 0.005
    __L_contrast_ratio_d = 1000
    __L_blk_d = None
    __gamma_d = 2.2

    # maximum tested lux value for ambient light
    # this is used as an operational range (to guarantee min and max output range)
    # this may be required by some IQA metrics that rescale inputs to some range, for ex., assume 0-1 input
    __MAXIMUM_E_AMB = 30000

    def __init__(self,
                 device_profile_name=None,
                 L_max=None,
                 L_min=None,
                 L_blk=None,
                 L_contrast_ratio=None,
                 reflectivity=None,
                 gamma=None,
                 ):
        """
        :param device_profile_name: must be a profile file "{device_name}.json" present in profile folder

        If any of the parameters below are specified, they will overwrite the parameters given in the profile file.

        :param L_max:
            the highest display luminance value for a white pixel
        :param L_min:
            the lowest display luminance value for a white pixel
            note: Lmin is not the luminance of the black level (L_blk) of the display
        :param L_contrast_ratio:
            the ratio between the maximum and the black level display brightness
        :param gamma:
            display gamma
        :param reflectivity:
            default display reflectivity ratio (this depends on display surface)
            Examples:
            1. ITU-R BT.500-11 recommends 6% or 0.06 for common displays
            https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.500-11-200206-S!!PDF-E.pdf
            2. Rafal Mantiuk's work where this display model was introduced and HDR-VDP-2 source code
            recommends 1% or 0.01
        """

        # load default device profile if a profile name is not given
        if device_profile_name is None or not device_profile_name:
            device_profile_name = DisplayModel.__device_profile_name_d

        # read default params from .json profile file with display's characteristics
        with open(DM_PROFILES_PATH + device_profile_name + '.json', 'r') as device_profile_file:
            device_profile_dict = json.load(device_profile_file)

            self.L_max = device_profile_dict.pop(DisplayModel.__L_max_field, DisplayModel.__L_max_d)
            self.L_min = device_profile_dict.pop(DisplayModel.__L_min_field, DisplayModel.__L_min_d)
            self.L_blk = device_profile_dict.pop(DisplayModel.__L_blk_field, DisplayModel.__L_blk_d)
            self.L_contrast_ratio = device_profile_dict.pop(DisplayModel.__L_contrast_ratio_field,
                                                            DisplayModel.__L_contrast_ratio_d)
            self.reflectivity = device_profile_dict.pop(DisplayModel.__reflectivity_field,
                                                        DisplayModel.__reflectivity_d)
            self.y = device_profile_dict.pop(DisplayModel.__gamma_field, DisplayModel.__gamma_d)

        # overwrite with custom params, if specified
        unchanged_if_input_none = lambda a, b: a if b is None else b
        self.L_max = unchanged_if_input_none(self.L_max, L_max)
        self.L_min = unchanged_if_input_none(self.L_min, L_min)
        self.L_blk = unchanged_if_input_none(self.L_blk, L_blk)
        self.L_contrast_ratio = float(unchanged_if_input_none(self.L_contrast_ratio, L_contrast_ratio))
        self.reflectivity = unchanged_if_input_none(self.reflectivity, reflectivity)
        self.y = unchanged_if_input_none(self.y, gamma)

    def get_L_upper_bound(self):
        return self.L_max + self.get_L_refl(DisplayModel.__MAXIMUM_E_AMB, inject_reflection=True)

    def get_L_max(self, E_amb, use_display_dimming, L_max=None, use_luminance_only=False):
        # if L_max override is given as an input
        if L_max is not None:
            if isinstance(L_max, np.ndarray):
                if len(L_max.shape) < 3 and not use_luminance_only:
                    return L_max[..., np.newaxis]  # h x w x 1
                return L_max
            else:
                return float(L_max)

        # if display dimming is disabled, simply return maximum display L_max
        if not use_display_dimming:
            return self.L_max

        # compute ideal display L_max and clamp between allowed display parameters
        L_max_ideal = get_desired_L_max(E_amb)
        L_max = max(self.L_min, min(self.L_max, L_max_ideal))

        return L_max

    def _get_L_blk(self, L_max):
        if self.L_contrast_ratio is None or self.L_contrast_ratio <= 0:
            return self.L_blk
        else:
            return L_max / self.L_contrast_ratio

    def get_L_refl(self, E_amb, inject_reflection=True, use_luminance_only=False):
        if not inject_reflection:
            return 0
        E_refl = get_E_refl(E_amb, use_luminance_only)
        return self.reflectivity * E_refl / np.pi

    def display_simulation(
            self,
            V,  # input rgb/luminance values in range [0-1]
            E_amb,  # ambient illuminance in lux (single value or array)
            L_max=None,  # input display maximum luminance
            use_luminance_only=False,  # whether we want to output rgb or luminance only
            inject_reflection=True,  # whether reflections will be added
            use_display_dimming=True,  # whether auto-brightness is enabled to control screen max Lum
            ):
        E_amb = np.clip(E_amb, 0, DisplayModel.__MAXIMUM_E_AMB)
        L_max = self.get_L_max(E_amb, use_display_dimming, L_max, use_luminance_only)
        L_refl = self.get_L_refl(E_amb, inject_reflection, use_luminance_only)
        L_blk = self._get_L_blk(L_max)

        if use_luminance_only and len(V.shape) == 3:
            V = rgb2gray_matlab(V)
        else:
            L_refl = ensure3d(L_refl)

        L_d = np.power(V, self.y) * (L_max - L_blk) + L_blk

        return L_d + L_refl

    # full inverse of the display model equation
    # can toggle whether if the effect of reflections and screen dimming should be undone
    def display_simulation_inverse(
            self,
            L,
            E_amb,
            L_max=None,
            use_luminance_only=False,  # whether we want to output rgb or luminance only
            undo_reflection=False,  # whether reflections will be removed
            undo_display_lum_profile=True,  # whether screen dimming is undone
            undo_gamma=True,
            ):
        E_amb = np.clip(E_amb, 0, DisplayModel.__MAXIMUM_E_AMB)
        L_max = self.get_L_max(E_amb, undo_display_lum_profile, L_max)
        L_refl = self.get_L_refl(E_amb, undo_reflection, use_luminance_only)
        L_blk = self._get_L_blk(L_max)

        # if lmax is given and an array, use the MAXIMUM value of lmax as upper bound,
        # else the effect of having a diffuser is undone (it is not visible)
        if L_max is not None and isinstance(L_max, np.ndarray):
            L_max = L_max.max()

        if isinstance(L_blk, np.ndarray):
            L_blk = L_blk.min()

        if len(L.shape) == 3:
            L_refl = ensure3d(L_refl)

        V = np.maximum((L - L_blk - L_refl) / (L_max - L_blk), 0.)

        if undo_gamma:
            V = np.power(V, 1. / self.y)

        return V

    # simple inverse of the display function
    @staticmethod
    def display_simulation_inverse_simple(
            L,
            L_max,
            y=2.2
            ):
        # LRT display luminance profile
        return np.power(L / L_max, 1. / y)
