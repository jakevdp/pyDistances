# run python setup.py build_ext
#  in order to build cpp files from the pyx files

import os

import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

distmetrics = Extension("pyDistances.distmetrics",
                        ["distmetrics.pyx"])

brute_neighbors = Extension("pyDistances.brute_neighbors",
                            ["brute_neighbors.pyx"])

ball_tree = Extension("pyDistances.ball_tree",
                      ["ball_tree.pyx"])

setup(
    name='pyDistances',
    version='1.0',
    cmdclass={'build_ext': build_ext},
    ext_modules=[distmetrics, brute_neighbors, ball_tree],
    include_dirs=[numpy.get_include(),
                  os.path.join(numpy.get_include(), 'numpy')],
    packages=['pyDistances'],
    package_dir={'pyDistances': ''}
    )
