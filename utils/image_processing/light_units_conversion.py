import math


def watt2lumens(watts):
    """
    1.000 lumen/m2 (Lux)=1/683.0 W/m2=0.001464 W/m2 at 555nm light wavelength
    :param watts:
    :return:
    """
    return 683.0 * watts


def lumen2watt(lumens):
    return lumens / 683.0


def nits2lux(nits):
    return nits * math.pi


def lux2nits(lux):
    return lux / math.pi
