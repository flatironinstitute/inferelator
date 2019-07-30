import os
import sys

from setuptools import setup, find_packages

# Set py2 version ceilings
if sys.version_info[0] == 2:
    install_requires = ["numpy<=1.16.1", "scipy<=1.2.1", "pandas<=0.24.2", "scikit-learn<=0.20.0", "matplotlib<3.0"]
elif sys.version_info[0] == 3:
    install_requires = ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib"]
else:
    raise ValueError("Python isn't py2 or py3. What have you done.")


# Require coverage and nose for testing
tests_require = ["coverage", "nose"]

# Current Inferelator Version Number
version = "0.3.0"

# Release Date

setup(
    name="inferelator",
    version=version,
    description="Inferelator: Network Inference",
    url="https://github.com/flatironinstitute/inferelator",
    author="Chris Jackson",
    author_email="cj59@nyu.edu",
    maintainer="Chris Jackson",
    maintainer_email="cj59@nyu.edu",
    packages=find_packages(include=["inferelator", "inferelator.*"], exclude=["tests", "*.tests"]),
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite="nose.collector",
)
