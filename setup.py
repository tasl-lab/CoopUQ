#!/usr/bin/env python

import os
from setuptools import setup, find_packages
import setuptools

def get_version() -> str:
    init = open(os.path.join("onpolicy", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

setup(
    name="onpolicy",
    version=get_version(),
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
