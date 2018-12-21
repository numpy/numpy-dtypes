from distutils.core import setup, Extension

import numpy as np

ext_modules = [
    Extension(
        'npytypes.rational.rational',
        sources=['npytypes/rational/rational.c'],
        include_dirs=[np.get_include()]
    ),

    Extension(
        'npytypes.quaternion.numpy_quaternion',
        sources=[
            'npytypes/quaternion/quaternion.c',
            'npytypes/quaternion/numpy_quaternion.c',
        ],
        include_dirs=[
            np.get_include(),
        ],
        extra_compile_args=[
            '-std=c99'
        ]
    )
]

setup(
    name='npytypes',
    version='0.1.1',
    url='https://github.com/numpy/numpy-dtypes',
    description='NumPy type extensions',
    requires=[
        'numpy',
    ],
    packages=[
        'npytypes',
        'npytypes.quaternion',
        'npytypes.rational'
    ],
    ext_modules=ext_modules
)
