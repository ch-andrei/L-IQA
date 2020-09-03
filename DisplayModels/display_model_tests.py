from DisplayModels.display_model import *


# display model Examples
if __name__=="__main__":
    from utils.image_processing.image_tools import *
    img = imread("228620182_redesign.png", "footage/images/bulk/imgs", format_float=True, rescale=1)
    print('running with shape', img.shape)

    def plot_screen_dimming():
        dm = DisplayModel()

        illuminant_lux_profiles = np.arange(0, 1000, 1)

        L_maxs = [dm.get_L_max(lux, use_display_dimming=True) for lux in illuminant_lux_profiles]

        from matplotlib import pyplot as plt
        plt.plot(illuminant_lux_profiles, L_maxs, 'r-')
        plt.xlabel("Ambient illumination (lux)")
        plt.ylabel("Display maximum luminance (cd/m2)")
        plt.show()

    def plot_reflection():
        dm = DisplayModel()

        illuminant_lux_profiles = np.arange(0, 20000, 10)

        L_refls = [dm.get_L_refl(lux, inject_reflection=True) for lux in illuminant_lux_profiles]

        from matplotlib import pyplot as plt
        plt.plot(illuminant_lux_profiles, L_refls, 'r-')
        plt.xlabel("Ambient illumination (lux)")
        plt.ylabel("Reflection amount (cd/m2)")
        plt.show()

    def plot_combined():
        dm = DisplayModel()

        thresh = dm.adaptive_brightness_enabled_threshold
        illuminant_lux_profiles_dark = np.arange(0, thresh, 1)
        illuminant_lux_profiles_bright = np.arange(thresh, 25000, 100)

        L_maxs_dark = [dm.get_L_max(lux, use_display_dimming=True) for lux in illuminant_lux_profiles_dark]
        L_refls_dark = [dm.get_L_refl(lux, inject_reflection=True) for lux in illuminant_lux_profiles_dark]
        L_dark = np.array(L_maxs_dark) + np.array(L_refls_dark)

        L_bright = [dm.L_max + dm.get_L_refl(lux, inject_reflection=True)
                    for lux in illuminant_lux_profiles_bright]

        illuminant_lux_profiles_combined = np.concatenate((illuminant_lux_profiles_dark,
                                                                illuminant_lux_profiles_bright))
        L_combined = np.concatenate((L_dark, L_bright))

        from matplotlib import pyplot as plt
        plt.semilogx(illuminant_lux_profiles_combined, L_combined, 'r-')
        plt.xlabel("Ambient illumination (lux)")
        plt.ylabel("Display Luminance w/ Refl. and Dimming (cd/m2)")
        plt.show()

    def plot_luminance_histogram(img):
        dm = DisplayModel()

        from matplotlib import pyplot as plt
        from utils.misc.ColorIterator import ColorIterator
        colors = ColorIterator()

        color_ref = colors.next()
        color_test = colors.next()

        from histogram_analysis import log_histogram
        illuminant_lux_profiles = [2, 100, 250, 500, 1000, 10000, 25000]
        for illuminant_lux in illuminant_lux_profiles:
            print("Simulating for", illuminant_lux)

            img_L_ref = dm.display_simulation(img, illuminant_lux,
                                              use_luminance_only=True,
                                              inject_reflection=False,
                                              use_display_dimming=False,
                                              )
            img_L_test = dm.display_simulation(img, illuminant_lux,
                                               use_luminance_only=True,
                                               inject_reflection=True,
                                               use_display_dimming=True,
                                               )

            num_bins = 255
            remove_outliers = False
            remove_zeros = False
            thresh = 0.1
            hist_ref, bins_ref = log_histogram(img_L_ref, num_bins, remove_outliers=remove_outliers,
                                               remove_zero_bins=remove_zeros, zero_threshold=thresh)
            hist_test, bins_test = log_histogram(img_L_test, num_bins, remove_outliers=remove_outliers,
                                                 remove_zero_bins=remove_zeros, zero_threshold=thresh)

            plt.figure()
            plt.grid()

            def plot(hist, bins, color, switch, label):
                alpha_global = 0.75

                dot = 'x' if switch < 0 else 'o' if switch == 1 else '+'  # dot only
                line = '-' if switch < 0 else '--'  # line only

                markersize = 3 if dot == '+' else 1

                # plt.plot(bins[:-1], hist, dot, color=color, markersize=markersize+2, alpha=alpha_global)
                # plt.plot(bins[:-1], hist, line, color=color, alpha=1, label=label)
                plt.fill_between(bins[1:], [0 for _ in range(len(hist))], hist, color=color, alpha=1, label=label)

                plt.legend()

            plot(hist_ref, bins_ref, color_ref, switch=0, label="ref image")
            plot(hist_test, bins_test, color_test, switch=1, label="test image")

            plt.title("Luminance histograms for reference and test simulated images at {} lux".format(illuminant_lux))
            plt.xlabel("Luminance (cd/m2)")
            plt.ylabel("Log count")
            plt.show(block=False)

            # ymin, ymax = plt.ylim()
            # xmin, xmax = plt.xlim()
            # print(ymin, ymax)
            # print(xmin, xmax)
            # plt.ylim((0, ymax))
            # plt.xlim((0, xmax))

        plt.show()

    # plot_combined()
    plot_luminance_histogram(img)