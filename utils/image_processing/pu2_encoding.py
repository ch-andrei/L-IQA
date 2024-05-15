# 2018, Andrei Chubarau <andrei.chubarau@mail.mcgill.ca>
#
# Python version of the MATLAB code for perceptually uniform encoding as described in
# Aydin, T. O., Mantiuk, R., & Seidel, H.-P. (2008). Extending quality
# metrics to full luminance range images. Proceedings of SPIE (p. 68060B–10).
# SPIE. doi:10.1117/12.765095
#
# Original MATLAB code available at
# https://sourceforge.net/projects/hdrvdp/files/simple_metrics/
# pu2_encode.m

import numpy as np
from scipy.integrate import cumtrapz

# global variables, computed once upon first call for PU encoding)
pu2e_P_lut = None
pu2e_l_lut = None
# compute data bounds and range
pu2e_data_bounds = None
pu2e_data_range = None

# PU encoding allowed luminance range (from 10^-5 to 10^10 cd/m2 as specified in the original paper)
pu2e_l_min = -5
pu2e_l_max = 10


# LUT computation function
def compute_luts(num_points=2**12 + 1):
    global pu2e_P_lut, pu2e_l_lut, pu2e_l_min, pu2e_l_max, pu2e_data_bounds, pu2e_data_range

    if pu2e_P_lut is None or pu2e_l_lut is None:  # caching for better performance

        def build_jndspace_from_S(l, S):
            L = np.power(10, l)
            dL = np.zeros(len(L))

            for k in range(len(L)):
                thr = L[k] / S[k]

                # Different than in the paper because integration is done in the log
                # domain - requires substitution with a Jacobian determinant
                dL[k] = 1. / thr * L[k] * np.log(10)

            return cumtrapz(dL, l)

        def hdrvdp_joint_rod_cone_sens(la, csf_sa):
            """
            % Copyright (c) 2011, Rafal Mantiuk <mantiuk@gmail.com>

            % Permission to use, copy, modify, and/or distribute this software for any
            % purpose with or without fee is hereby granted, provided that the above
            % copyright notice and this permission notice appear in all copies.
            %
            % THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
            % WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
            % MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
            % ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
            % WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
            % ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
            % OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

            :param la:
            :param csf_sa:
            :return:
            """

            cvi_sens_drop = csf_sa[1]  # in the paper - p6
            cvi_trans_slope = csf_sa[2]  # in the paper - p7
            cvi_low_slope = csf_sa[3]  # in the paper - p8

            return csf_sa[0] * np.power(np.power(cvi_sens_drop / la, cvi_trans_slope) + 1,
                                        -cvi_low_slope)

        # initialize LUTs

        csf_sa = np.array([30.162, 4.0627, 1.6596, 0.2712])
        pu2e_l_lut = np.linspace(pu2e_l_min, pu2e_l_max, num_points)

        S = hdrvdp_joint_rod_cone_sens(np.power(10., pu2e_l_lut), csf_sa)

        pu2e_P_lut = build_jndspace_from_S(pu2e_l_lut, S)
        pu2e_l_lut = pu2e_l_lut[:-1]  # remove last entry (numpy linspace makes one extra value)

        # initialize data bounds

        pu2e_data_bounds = (pu2_encode(10 ** pu2e_l_min), pu2_encode(10 ** pu2e_l_max))
        pu2e_data_range = pu2e_data_bounds[1] - pu2e_data_bounds[0]

        print("Initialized PU encoding.")


def pu2_encode(L):
    # Original Copyright Notice:
    """
    Perceptually uniform luminance encoding using the CSF from HDR-VDP-2

    P = pu2_encode( L )

    Transforms absolute luminance values L into approximately perceptually
    uniform values P.

    This is meant to be used with display-referred quality metrics - the
    image values must correspond to the luminance emitted from the target
    HDR display.

    This is an improved encoding described in detail in the paper:

    Aydin, T. O., Mantiuk, R., & Seidel, H.-P. (2008). Extending quality
    metrics to full luminance range images. Proceedings of SPIE (p. 68060B–10).
    SPIE. doi:10.1117/12.765095

    Note that the P-values can be negative or greater than 255. Most metrics
    can deal with such values.

    Copyright (c) 2014, Rafal Mantiuk <mantiuk@gmail.com>
    """

    global pu2e_P_lut, pu2e_l_lut, pu2e_l_min, pu2e_l_max

    l = np.log10(np.clip(L, 10 ** pu2e_l_min, 10 ** pu2e_l_max))

    pu_l = 31.9270
    pu_h = 149.9244

    return 255. * (np.interp(l, pu2e_l_lut, pu2e_P_lut) - pu_l) / (pu_h - pu_l)


def pu2_encode_offset(L):
    global pu2e_data_bounds

    return pu2_encode(L) - pu2e_data_bounds[0]


# initialize PU encoding variables
compute_luts()


# test
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    print(pu2e_data_bounds[0])
    print(pu2e_data_bounds[1])
    print(pu2e_data_range)
    print('input min', 10**pu2e_l_min, 'output', pu2_encode(0))
    print('input max', 10**pu2e_l_max, 'output', pu2_encode(10**pu2e_l_max))
    print('offset input min', 10**pu2e_l_min, 'output', pu2_encode_offset(0))
    print('offset input max', 10**pu2e_l_max, 'output', pu2_encode_offset(10**pu2e_l_max))
    plt.plot(pu2e_l_lut, pu2e_P_lut)
    plt.show()
