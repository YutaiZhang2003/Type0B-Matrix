"""Physical VV components relative to the saved fixed-parity chiral banks.

The long-NS-ground-state limit is a sphere correlator with words
M=(P,G,G,P) on the left and P=(P,P,P,P) on the right. For homogeneous
form f, the left internal parity is r=f xor 1. The established sphere
component phase is (-1)^r, while the saved necklace pairing is (-1)^f.
Their ratio is -1 for BOTH forms, independent of weights or levels.
PP has ratio +1. This is an external-component conversion, and therefore
the same multiplier applies to every torus spin structure. It is not a
modification of the recursive coefficients or a fitted collision sign.
"""
import numpy as np

CONVENTION = 'spin23-vv-physical-sphere-components-v2'


def component_conversion(word):
    word=tuple(word)
    if word==(1,1):return -1
    if word==(0,0):return 1
    raise ValueError('the VV conversion is defined for GG and PP only')


def collision_ward_coefficients(omega,h):
    """[... , even/odd bridge family, GG/time-fermion PCO component].

    Both PCO terms use the SAME ordered local coordinate delta=z0-z1.
    GG carries (2*h_E-h) or (2*h_E+h-1/2), and the signed time momenta
    multiply to -omega**2. Thus the physical sums are 1-h and h+1/2.
    """
    h=np.asarray(h,complex);omega=complex(omega)
    if not np.isfinite(h).all() or not np.isfinite(omega):
        raise ValueError('finite external energy and bridge weights required')
    out=np.empty(h.shape+(2,2),complex)
    out[...,0,0]=1+omega**2-h
    out[...,1,0]=h+.5+omega**2
    out[..., :,1]=-omega**2
    return out
