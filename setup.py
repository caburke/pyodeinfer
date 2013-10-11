# Set up for the odeinfer package
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("popmcmc_test", ["popmcmc_test.c"])]
)