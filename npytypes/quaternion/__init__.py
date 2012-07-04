import numpy as np

from .numpy_quaternion import quaternion
from .info import __doc__

__all__ = ['quaternion']

if np.__dict__.get('quaternion') is not None:
    raise RuntimeError('The NumPy package already has a quaternion type')

np.quaternion = quaternion
np.typeDict['quaternion'] = np.dtype(quaternion)
