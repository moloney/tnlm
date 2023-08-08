import os
import sys

from Cython.Build import cythonize
from distutils.command.build_ext import build_ext

# use cythonize to build the extensions
modules = ["tnlm/_tnlm.pyx",]

extensions = cythonize(modules)

def build(setup_kwargs):
    """Needed for the poetry building interface."""

    setup_kwargs.update({
        'ext_modules' : extensions,
        'cmdclass': {'build_ext': build_ext}
    })
