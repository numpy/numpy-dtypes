from distutils.core import setup, Extension
import numpy as np

ext_modules = []

ext = Extension('npytypes.rational.rational',
                sources=['npytypes/rational/rational.c'],
                include_dirs=[np.get_include()])
ext_modules.append(ext)

ext = Extension('npytypes.quaternion.numpy_quaternion',
                sources=['npytypes/quaternion/quaternion.c',
                         'npytypes/quaternion/numpy_quaternion.c'],
                include_dirs=[np.get_include()],
                extra_compile_args=['-std=c99'])
ext_modules.append(ext)

setup(name='npytypes',
      version='0.1',
      description='NumPy type extensions',
      packages=['npytypes',
                'npytypes.quaternion',
                'npytypes.rational'
                ],
      ext_modules=ext_modules)
