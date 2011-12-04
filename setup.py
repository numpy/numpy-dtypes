#!/usr/bin/env python

from distutils.core import setup, Extension
from numpy.distutils.system_info import get_info

module = Extension('rational',
                   sources = ['rational.cpp'],
                   # extra_compile_args = ['-g'],
                   include_dirs = get_info('numpy')['include_dirs'])

setup(name = 'rational',
      version = '1.0',
      description = 'Fixed precision rational arithmetic',
      ext_modules = [module])
