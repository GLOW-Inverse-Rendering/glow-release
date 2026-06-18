import numpy as np
def srgb_to_linsrgb (srgb):
    """Convert sRGB values to physically linear ones. The transformation is
       uniform in RGB, so *srgb* can be of any shape.

       *srgb* values should range between 0 and 1, inclusively.

    """
    gamma = ((srgb + 0.055) / 1.055)**2.4
    scale = srgb / 12.92
    return np.where (srgb > 0.04045, gamma, scale)

srgb2lin = srgb_to_linsrgb

def lin2srgb(lin):
    s1 = 1.055 * (np.power(lin, (1.0 / 2.4))) - 0.055
    s2 = 12.92 * lin
    s = np.where(lin > 0.0031308, s1, s2)
    return np.minimum(s, 1.0)
