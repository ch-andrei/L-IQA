import numpy as np
import json

from utils.image_processing.image_tools import ensure3d
from utils.image_processing.color_spaces import srgb2lum, srgb2rgb, rgb2srgb


class DisplayModel:
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

    Lrefl = k * E_amb / np.pi

        k is the reflectivity constant,
        E_amb the ambient illumination level in lux

    """
    def __init__(self,
                 device_name=None,
                 L_max=None,
                 L_min=None,
                 contrast_ratio=None,
                 gamma=2.2,
                 reflectivity=0.01
                 ):
        """

        :param device_name:
            profile folder must contain a file named "{device_name}.json" where device_name is .json filename
        :param L_max:
            the highest display luminance value for a white pixel
        :param L_min:
            the lowest display luminance value for a white pixel
            note: Lmin is not the luminance of the black level (L_blk) of the display
        :param contrast_ratio:
            the ratio between the maximum and the black level display brightness
        :param gamma:
            display gamma
        :param reflectivity:
            default display reflectivity ratio (this depends on display surface)
            Examples:
            1. ITU-R BT.500-11 recommends 6% or 0.06 for common displays
            https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.500-11-200206-S!!PDF-E.pdf
            2. Rafal Mantiuk's work where this display model was introduced and HDR-VDP-2 source code
            recommend 1% or 0.01
        """

        # CONSTANTS AND CONSTRAINTS

        # maximum tested lux value for ambient light
        # this is used as an operational range (to guarantee min and max output range)
        # this may be required by some IQA metrics that rescale inputs to some range, for ex., assume 0-1 input
        self.maximum_E_amb = 100000

        # path to .json profiles
        profiles_path = 'DisplayModels/profiles/'
        ambient_profile = 'ambient_profile.json'
        # constant namefields to fetch from .json profile
        L_max_field = "L_max"
        L_min_field = "L_min"
        L_blk = "L_blk"
        L_contrast_ratio = "L_contrast_ratio"
        lut_ambient_lux_field = "lut_ambient_lux"
        lut_target_luminance_field = "lut_target_luminance"
        reflectivity_field = "k_reflectivity"

        # default device name
        if device_name is None or not device_name:
            device_name = "d500"

        # CONSTANTS END

        # display gamma
        self.y = gamma

        # read .json profile file with display's luminance characteristics and display dimming profile
        with open(profiles_path + device_name + '.json', 'r') as device_profile_file, \
                open(profiles_path + ambient_profile, 'r') as ambient_profile_file:
            device_profile = json.load(device_profile_file)
            ambient_profile = json.load(ambient_profile_file)

            # display maximum and minimum luminance level (white point luminance)
            # Lmax occurs at maximum display brightness, Lmin at lowest
            if L_max is None:
                self.L_max = device_profile[L_max_field]
            else:
                self.L_max = float(L_max)

            if L_min is None:
                self.L_min = device_profile[L_min_field]
            else:
                self.L_min = float(L_min)

            try:
                self.k = device_profile[reflectivity_field]
            except KeyError:
                self.k = reflectivity

            if contrast_ratio is None:
                # not all profiles have this defined
                try:
                    self.L_contrast_ratio = device_profile[L_contrast_ratio]
                except KeyError:
                    try:
                        # use constant black level
                        self.L_contrast_ratio = None
                        self.L_blk = device_profile[L_blk]
                    except KeyError:
                        # default to the ratio between luminance values for max and min level
                        self.L_contrast_ratio = 100
            else:
                self.L_contrast_ratio = float(contrast_ratio)

            # desired display adaptive brightness profile as LUT
            self.lut_ambient_lux = np.array(ambient_profile[lut_ambient_lux_field])
            self.lut_lum_target = np.array(ambient_profile[lut_target_luminance_field])

        # maximum ambient condition when adaptive brightness is enabled
        self.adaptive_brightness_enabled_threshold = np.interp(self.L_max, self.lut_lum_target, self.lut_ambient_lux)

    def get_L_upper_bound(self):
        return self.L_max + self.get_L_refl(self.maximum_E_amb, inject_reflection=True)

    def get_L_max(self, E_amb, use_display_dimming):
        if not use_display_dimming:
            return self.L_max

        L_max = np.interp(E_amb, self.lut_ambient_lux, self.lut_lum_target)  # theoretical desired L_max value

        return max(self.L_min, min(self.L_max, L_max))

    def _get_L_blk(self, L_max):
        # return L_max / self.L_contrast_ratio + self._get_L_refl(E_amb, inject_reflection)
        return self.L_blk if self.L_contrast_ratio is None else L_max / self.L_contrast_ratio

    def get_L_refl(self, E_amb, inject_reflection=True):
        if not inject_reflection:
            return 0

        return self.k * E_amb / np.pi

    def _get_E_refl(self, E_amb, illumination_map, illumination_map_weight_mode, use_luminance_only=True):
        if illumination_map is None:
            return E_amb

        if illumination_map_weight_mode is None:
            illumination_map_weight_mode = 'mean'

        illumination_map_func = np.mean if illumination_map_weight_mode == 'mean' else np.max

        if len(illumination_map.shape) == 3:
            lum_map = srgb2lum(illumination_map)

            if use_luminance_only:
                return E_amb * lum_map / lum_map.mean()
            else:
                lum_map_weighted = ensure3d(lum_map) * illumination_map
                lum_map_weighted = lum_map_weighted / illumination_map_func(lum_map_weighted)
                return E_amb * lum_map_weighted

        return E_amb * illumination_map / illumination_map_func(illumination_map)

    def display_simulation(
            self,
            data,  # input rgb/luminance values in range [0-1]
            E_amb,  # ambient illuminance in lux (single value or array)
            illumination_map=None,
            illumination_map_weight_mode=None,  # select from [None, 'mean', 'max']
            use_luminance_only=True,  # whether we want to output rgb or luminance only
            inject_reflection=True,  # whether reflections will be added
            use_display_dimming=True,  # whether auto-brightness is enabled to control screen max Lum
            ):
        E_amb = np.clip(E_amb, 0, self.maximum_E_amb)
        L_max = self.get_L_max(E_amb, use_display_dimming)
        E_refl = self._get_E_refl(E_amb, illumination_map, illumination_map_weight_mode, use_luminance_only)
        L_refl = self.get_L_refl(E_refl, inject_reflection)
        L_blk = self._get_L_blk(L_max)

        if use_luminance_only and len(data.shape) == 3:
            L = srgb2lum(data)
        else:
            L = srgb2rgb(data)
            L_refl = ensure3d(L_refl)

        return np.power(L, self.y) * (L_max - L_blk) + L_blk + L_refl

    # full inverse of the display model equation
    # can toggle whether if the effect of reflections and screen dimming should be undone
    def display_simulation_inverse(
            self,
            L,
            E_amb,
            illumination_map=None,
            illumination_map_weight_mode=None,
            use_luminance_only=True,  # whether we want to output rgb or luminance only
            undo_reflection=True,  # whether reflections will be removed
            undo_display_lum_profile=True,  # whether screen dimming is undone
            undo_gamma=True
            ):
        E_amb = np.clip(E_amb, 0, self.maximum_E_amb)
        L_max = self.get_L_max(E_amb, undo_display_lum_profile)
        E_refl = self._get_E_refl(E_amb, illumination_map, illumination_map_weight_mode, use_luminance_only)
        L_refl = self.get_L_refl(E_refl, undo_reflection)
        L_blk = self._get_L_blk(L_max)

        if len(L.shape) == 3:
            L_refl = ensure3d(L_refl)

        L_out = np.power(np.maximum((L - L_blk - L_refl) / (L_max - L_blk), 0.), 1. / self.y)

        if undo_gamma:
            L_out = rgb2srgb(ensure3d(L_out))
        else:
            L_out = (rgb2srgb(ensure3d(L_out)))[..., 0]  # TODO: why take R channel?

        return L_out

    # simple inverse of the display function
    @staticmethod
    def display_simulation_inverse_simple(
            L,
            L_max,
            y=2.2
            ):
        # LRT display luminance profile
        return np.power(L / L_max, 1. / y)
