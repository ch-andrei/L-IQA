from utils.image_processing.image_tools import *
from iqa_metrics.iqa_tool import *

import cv2


def main():
    iqa_tool = IqaTool()

    lux_ref = 0  # reference illumination condition in lux
    lux_tests = [1000, 5000, 20000]  # test illumination conditions in lux

    path = 'images/'
    filename = "airport_inside_0549_crop"

    illum_map_path = 'images/'
    illum_map_file = '20200424_095910-2500lux-blurry'

    print('Running for image', filename)

    img_ref = imread_unknown_extension(filename, path, format_float=True)

    img_illum = imread_unknown_extension(illum_map_file, illum_map_path,
                                         format_float=True,
                                         rescale_if_too_big=False)

    img_illum = cv2.flip(img_illum, 1)  # mirror the reflection image horizontally

    illumination_map = resize_keep_aspect_ratio(
        img_illum,
        resolution=img_ref.shape[:2],
        zoom=True
    )

    # NOTE: if want to use uniform reflection, uncomment the following line
    # illumination_map = None

    # add random noise to image
    noise_magnitude = 0.1
    noise = np.random.rand(*img_ref.shape)
    img_test = img_ref + noise_magnitude * (-1 + 2 * noise)
    img_test = np.clip(img_test, 0, 1)

    for lux_test in lux_tests:
        sim_params_ref = new_simul_params(illuminant=lux_ref, illumination_map=illumination_map,
                                          apply_screen_dimming=False, use_luminance_only=False)
        sim_params_test = new_simul_params(illuminant=lux_test, illumination_map=illumination_map,
                                           apply_screen_dimming=False, use_luminance_only=False)

        Qs = iqa_tool.compute_iqa_custom(img_ref, img_test, sim_params_ref, sim_params_test)

        print('Test condition {} lux:'.format(lux_test))
        print(Qs)


if __name__ == "__main__":
    main()
