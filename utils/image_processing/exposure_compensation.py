import math
from utils.image_processing.light_units_conversion import *
from utils.image_processing.image_tools import *


def get_standard_output_exposure_l_avg(aperture2_to_speed_ratio,
                                       iso):
    return (1000.0 / 65.0) * aperture2_to_speed_ratio / iso


# Get an exposure using the Standard Output Sensitivity method.
# Accepts an additional parameter of the target middle grey.
def get_standard_output_exposure(aperture2_to_speed_ratio,
                                 iso,
                                 middle_grey=0.18):
    l_avg = get_standard_output_exposure_l_avg(aperture2_to_speed_ratio, iso)
    return middle_grey / l_avg


# Given an aperture, shutter speed, and exposure value compute the required ISO value
def compute_iso(aperture2_to_speed_ratio, ev):
    return aperture2_to_speed_ratio * 100.0 / math.pow(2.0, ev)


# Given the camera settings compute the current exposure value
def compute_ev(aperture2_to_speed_ratio, iso):
    return math.log2(aperture2_to_speed_ratio * 100.0 / iso)


def compute_diff_ev(target_ev,
                    aperture2_to_speed_ratio,
                    iso):
    # Figure out how far we are from the target exposure value
    return target_ev - compute_ev(aperture2_to_speed_ratio, iso)


# Using the light metering equation compute the target exposure value
# the relationship is given by the exposure equation prescribed by ISO 2720:1974
# N^2 / t = L S / K
# N is aperture, t is shutter speed, L is average scene luminance, S is ISO, K is a light meter calibration constant
# exposure is the log2 of aperture2_to_speed_ratio
def get_target_profile(average_luminance):
    # target EV at 100ISO
    K = 12.5
    iso = 100
    aperture2_to_speed_ratio = average_luminance * iso / K # aperture2_to_shutter_speed_ratio
    return math.log2(aperture2_to_speed_ratio)


def exposure_compensation(img,
                          illuminant_lux,  # ambient illumination strength in lux
                          luminance_target=300,  # will compensate for this luminance, in cd/m2
                          luminance_display=300,  # the luminance of the display cd/m2
                          ):
    """
    naive exposure compensation algorithm
    :param img:
    :param illuminant_lux:
    :param luminance_target:
    :param luminance_display:
    :return:
    """
    target_ev = get_target_profile(luminance_target)

    lum_bg = lux2nits(illuminant_lux) # bg luminance cd / m2
    lum_avg_actual = luminance_display + lum_bg

    current_ev = get_target_profile(lum_avg_actual)

    ev_diff = current_ev - target_ev

    compensation = 2 ** -ev_diff

    # apply exposure compensation
    img_compensated = img * compensation


    return img_compensated


def gamma_correction(img, gamma=2.2, epsilon=1e-9):
    max_val = img.max()

    if max_val < epsilon:
        return img

    img /= max_val
    img = np.power(img, 1.0 / gamma)
    img *= max_val  # TODO: correct this, not real gamma -> bad scaling with dividing and multiplying by max; verify

    return img


def hdr_compensation(img, illuminant_lux,
                     apply_gamma=True,
                     apply_exposure=True,
                     apply_normalize=False,
                     apply_clip=True):
    if apply_gamma:
        # apply gamma correction
        img = gamma_correction(img)

    if apply_exposure:
        # apply exposure compensation
        img = exposure_compensation(img, illuminant_lux)

    if apply_normalize:
        img = img / img.max()

    if apply_clip:
        # clamp the range to 0-1
        img = np.clip(img, 0, 1)

    return img
