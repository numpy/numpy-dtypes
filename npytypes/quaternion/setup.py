#!/usr/bin/env python

from distutils.core import setup, Extension

import numpy as np
from distutils.errors import DistutilsError

if np.__dict__.get('quaternion') is not None:
    raise DistutilsError('The target NumPy already has a quaternion type')

quat_ext = Extension('numpy_quaternion',
                     sources=['quaternion.c',
                              'numpy_quaternion.c'],
                     include_dirs=[np.get_include()],
                     extra_compile_args=['-std=c99'])

setup(name='quaternion',
      version='1.0',
      description='Quaternion NumPy dtype',
      ext_modules=[quat_ext])
