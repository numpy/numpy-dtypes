#!/usr/bin/env python

from distutils.core import setup, Extension
# from numpy.distutils.system_info import get_info
import numpy as np

module = Extension('rational',
                   sources = ['rational.c'],
                   # extra_compile_args = ['-g'],
                   include_dirs = [np.get_include()])

setup(name = 'rational',
      version = '1.0',
      description = 'Fixed precision rational arithmetic',
      ext_modules = [module])
