import numpy as np

from utils.misc.miscelaneous import lerp

PU21_TYPE_BANDING = 0
PU21_TYPE_BANDING_GLARE = 1
PU21_TYPE_PEAKS = 2
PU21_TYPE_PEAKS_GLARE = 3


class PUTransform(object):
    """
    Transform absolute linear luminance values to/from the perceptually
    uniform (PU) space. This class is intended for adapting image quality
    metrics to operate on HDR content.

    The derivation of the PU21 encoding is explained in the paper:

    R. K. Mantiuk and M. Azimi, "PU21: A novel perceptually uniform encoding for adapting existing quality metrics
    for HDR," 2021 Picture Coding Symposium (PCS), 2021, pp. 1-5, doi: 10.1109/PCS50896.2021.9477471.

    Aydin TO, Mantiuk R, Seidel H-P.
    Extending quality metrics to full luminance range images.
    In: Human Vision and Electronic Imaging. Spie 2008. no. 68060B.
    DOI: 10.1117/12.765095

    The original MATLAB implementation is ported to Python and modified for realistic display systems.

    Modified by Andrei Chubarau to linearly encode luminance values less than 0.005cd/m2 using the slope of
    the PU-encoded values at L=0.005cd/m2 (the original minimum supported luminance level)
    NOTE:
        The original PU21 encoding supports a luminance range of 0.005-10000cd/m2.
        We optionally modify L_min to be 0cd/m2. Since PU21 for L<0.005cd/m2 may give negative values,
        we offset the encoded values upward by P_min (small negative value output when input L = 0cd/m2).
        This avoids negative encoded values and allows us to apply PU encoding to the extreme dark condition.
        This change is minor and does not alter the overall shape/format of PU encoding since P_min is small.
    """

    # PU21 parameters in order: [PU21_TYPE_BANDING, PU21_TYPE_BANDING_GLARE, PU21_TYPE_PEAKS, PU21_TYPE_PEAKS_GLARE]
    __par = [
        [1.070275272, 0.4088273932, 0.153224308, 0.2520326168, 1.063512885, 1.14115047, 521.4527484],
        [0.353487901, 0.3734658629, 8.2770492e-05, 0.9062562627, 0.0915030316, 0.90995172, 596.3148142],
        [1.043882782, 0.6459495343, 0.3194584211, 0.374025247, 1.114783422, 1.095360363, 384.9217577],
        [816.885024, 1479.463946, 0.001253215609, 0.9329636822, 0.06746643971, 1.573435413, 419.6006374],
    ]

    def __init__(self,
                 encoding_type: int = PU21_TYPE_BANDING_GLARE,
                 normalize=True,
                 normalize_range_srgb=False,
                 L_min_0=True,
                 ):
        """
        :param encoding_type:
        :param normalize: toggle for rescaling the output to range ~[0.0, 1.0]
        :param normalize_range_srgb: when true, PU-encoded value 255 (instead of P_max) will map to 1.0.
            Output range then is not ~[0.0, 1.0] but ~[0.0, 2.5] with sRGB@100cd/m2 encoded as approximately [0.0, 1.0]
        :param L_min_0: extend the minimum L range to 0.0cd/m2 instead of 0.005cd/m2
        """
        super().__init__()

        self.L_min = 0.0 if L_min_0 else 0.005
        self.L_max = 10000

        if encoding_type not in [PU21_TYPE_BANDING, PU21_TYPE_BANDING_GLARE, PU21_TYPE_PEAKS, PU21_TYPE_PEAKS_GLARE]:
            raise ValueError("Unsupported PU21 encoding type.")

        self.encoding_type = encoding_type
        self.par = self.__par[encoding_type]

        # NOTE: start with normalization/L_min_0 disabled to compute scaling parameters using the original PU scale
        self.L_min_0 = False
        self.normalize = False
        if L_min_0:
            # compute slope at L=0.005cd/m2 and the corresponding y-intersect given linear equation of form y=ax+b
            h = 1e-5
            L_005 = 0.005
            self.P_005_m = (self.pu_encode(L_005 + h) - self.pu_encode(L_005)) / h  # slope
            self.P_005_y = self.pu_encode(L_005) - L_005 * self.P_005_m  # y-intersect
        # set the final L_min_0 parameter
        self.L_min_0 = L_min_0

        # compute upper range for normalization
        # use PU-encoded value at L=10000cd/m2 or at typical CRT display luminance (SDR) of 100cd/m2
        self.P_max = self.pu_encode(100 if normalize_range_srgb else 10000)
        # compute lower range for normalization
        self.P_min = self.pu_encode(self.L_min)

        # set the final normalization params
        self.normalize = normalize
        self.normalize_range_srgb = normalize_range_srgb

    def __call__(self, Y):
        return self.pu_encode_scale(Y)

    @staticmethod
    def pu_encode_poly(Y, p):
        return p[6] * (((p[0] + p[1] * Y ** p[3]) / (1 + p[2] * Y ** p[3])) ** p[4] - p[5])

    def pu_encode(self, Y):
        """
        Convert from linear (optical) values Y to encoded (electronic) values V
        Y should be scaled in the absolute units (nits, cd/m^2).
        """
        Y = np.clip(Y, self.L_min, self.L_max)
        V = self.pu_encode_poly(Y, self.par)

        if self.L_min_0:
            V_005 = self.P_005_m * Y + self.P_005_y  # linear extension for 0.0 < L < 0.005 cd/m2
            V = lerp(V_005, V, 0.005 < Y)

        return V

    def pu_encode_scale(self, Y):
        V = self.pu_encode(Y)

        if self.normalize or self.L_min_0:
            V -= self.P_min

        if self.normalize:
            V /= (self.P_max - self.P_min)

        return V


if __name__ == "__main__":
    pu = PUTransform(encoding_type=PU21_TYPE_BANDING_GLARE, normalize=True, normalize_range_srgb=False, L_min_0=True)

    print("0->", pu(0))
    print("0.005->", pu(0.005))
    print("10000->", pu(10000))
