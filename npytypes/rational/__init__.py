import numpy as np

from npytypes.rational.rational import denominator, gcd, lcm, numerator, rational
from npytypes.rational.info import __doc__

__all__ = ['denominator', 'gcd', 'lcm', 'numerator', 'rational']

if np.__dict__.get('rational') is not None:
    raise RuntimeError('The NumPy package already has a rational type')

np.rational = rational
np.typeDict['rational'] = np.dtype(rational)
