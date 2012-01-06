# run python setup.py build_ext
#  in order to build cpp files from the pyx files

import os

import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext = Extension("distmetrics",
                ["distmetrics.pyx"]
                )

setup(cmdclass = {'build_ext': build_ext},
      name='distmetrics',
      version='1.0',
      ext_modules=[ext],
      )
