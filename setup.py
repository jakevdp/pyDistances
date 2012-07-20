# run python setup.py build_ext
#  in order to build cpp files from the pyx files

import os

import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

distmetrics = Extension("distmetrics",
                        ["distmetrics.pyx"])

brute_neighbors = Extension("brute_neighbors",
                            ["brute_neighbors.pyx"])

ball_tree = Extension("ball_tree",
                      ["ball_tree.pyx"])

setup(cmdclass = {'build_ext': build_ext},
      name='distmetrics',
      version='1.0',
      ext_modules=[distmetrics],
      include_dirs=[numpy.get_include(),
                    os.path.join(numpy.get_include(), 'numpy')]
      )

setup(cmdclass = {'build_ext': build_ext},
      name='brute_neighbors',
      version='1.0',
      ext_modules=[brute_neighbors],
      include_dirs=[numpy.get_include(),
                    os.path.join(numpy.get_include(), 'numpy')]
      )

setup(cmdclass = {'build_ext': build_ext},
      name='ball_tree',
      version='1.0',
      ext_modules=[ball_tree],
      include_dirs=[numpy.get_include(),
                    os.path.join(numpy.get_include(), 'numpy')]
      )
