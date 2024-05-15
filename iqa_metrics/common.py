import numpy as np

from utils.image_processing.color_spaces import rgb2lum


# possible data formats with approximate ranges (ranges can differ)
DATA_FORMAT_SRGB = 0  # ~0-255
DATA_FORMAT_LUM = 1  # ~0-10000 cd/m2
DATA_FORMAT_PU = 2  # ~0-600


def kwargs_get_data_params(**kwargs):
    data_range = kwargs.pop('data_range', (0., 255.))
    data_format = kwargs.pop('data_format', DATA_FORMAT_SRGB)
    return data_range, data_format


def normalize(img, data_range):
    vmin, vmax = data_range
    return (np.array(img, float) - vmin) / (vmax - vmin)


def ensure_single_channel_lum(img):
    if len(img.shape) > 2:
        img = rgb2lum(img)
    return img


def transform_data_255(img, data_range, data_format):
    """
    Transforms img into values comparable to 0-255 sRGB data.
    When input is sRGB or PU-encoded, simply returns the input.
    When input is Luminance, rescales input from 0-10000cd/m2 to 0-255.
    :param img:
    :param data_range:
    :param data_format:
    :return:
    """

    if data_format == DATA_FORMAT_SRGB:
        # if input is sRGB, ensure range is 0-255 and return the img
        if data_range[0] == 0. and data_range[1] == 255.:
            return img
        else:
            return 255 * normalize(img, data_range)

    if data_format == DATA_FORMAT_PU:
        # if input is PU-encoded, simply return the img with the PU scaling range (this may exceed 255)
        return img

    if data_format == DATA_FORMAT_LUM:
        # if input is luminance, normalize from 0-10000 cd/m2 to 0-255
        return 255 * normalize(img, data_range)

    else:
        raise ValueError(f"Unsupported data format.")
