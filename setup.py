import os
from setuptools import setup
import sys

if sys.version_info[0] == 2:
    install_requires = ["numpy", "scipy<=1.2.1", "pandas", "scikit-learn<=0.20.0", "matplotlib<3.0"]
elif sys.version_info[0] == 3:
    install_requires = ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib"]

tests_require = ["coverage", "nose"]

base_dir = os.path.dirname(os.path.abspath(__file__))

version = "0.2"

long_description = "\n\n".join([open(os.path.join(base_dir, "README.md"), "r").read()])

setup(
    name = "inferelator",
    version = version,
    description = "Inferelator: Network Inference",
    long_description=long_description,
    url = "https://github.com/flatironinstitute/inferelator",
    author = "Aaron Watters",
    author_email = "awatters@simonsfoundation.org",
    maintainer = "Aaron Watters",
    maintainer_email = "awatters@simonsfoundation.org",
    packages = ["inferelator"],
    zip_safe = False,
    install_requires = install_requires,
    tests_require = tests_require,
    test_suite = "nose.collector",
)
