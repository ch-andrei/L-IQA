from utils.image_processing.image_tools import *
from DisplayModels.display_model_simul import DisplayDegradationModel, new_simul_params

import cv2
import time

from pathlib import Path


def main():
    do_write = True

    ddm = DisplayDegradationModel("s10")  # Samsung Galaxy s10 display profile

    lux_ref = 0  # reference illumination condition in lux
    lux_tests = [1000, 5000, 20000]  # test illumination conditions in lux

    path = 'images/'
    filename = "airport_inside_0549_crop"

    print('Running for image', filename)

    illum_map_path = 'images/'
    illum_map_file = '20200424_095910-2500lux-blurry'

    output_map_path = "images/reflection/{}".format(int(time.time()))
    print('Writing to', output_map_path)

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

    if do_write:
        Path(output_map_path).mkdir(parents=True, exist_ok=True)

        imwrite('reference.jpg', output_map_path, img_ref)
        imwrite('illumination_map.jpg', output_map_path, illumination_map)

    for lux_test in lux_tests:
        print('Computing for condition {} lux'.format(lux_test))

        # will simulate two images:
        # 1. the reference image in reference conditions
        # 2. the reference image in test conditions
        sim_params_ref = new_simul_params(illuminant=lux_ref, illumination_map=illumination_map,
                                          apply_screen_dimming=False, use_luminance_only=False)
        sim_params_test = new_simul_params(illuminant=lux_test, illumination_map=illumination_map,
                                           apply_screen_dimming=False, use_luminance_only=False)

        rgb1s = ddm.simulate_displays_rgb(img_ref, sim_params_ref)
        rgb2s = ddm.simulate_displays_rgb(img_ref, sim_params_test)

        if do_write:
            imwrite('ref-s-{}.jpg'.format(lux_ref), output_map_path, rgb1s)
            imwrite('test-s-{}.jpg'.format(lux_test), output_map_path, rgb2s)


if __name__ == "__main__":
    main()
