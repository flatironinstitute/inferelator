import os
from setuptools import setup
import sys

# Set py2 version ceilings
if sys.version_info[0] == 2:
    install_requires = ["numpy", "scipy<=1.2.1", "pandas<=0.24.2", "scikit-learn<=0.20.0", "matplotlib<3.0"]
elif sys.version_info[0] == 3:
    install_requires = ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib"]
else:
    raise ValueError("Python isn't py2 or py3. What have you done.")

tests_require = ["coverage", "nose"]

base_dir = os.path.dirname(os.path.abspath(__file__))

version = "0.3"

long_description = "\n\n".join([open(os.path.join(base_dir, "README.md"), "r").read()])

setup(
    name = "inferelator",
    version = version,
    description = "Inferelator: Network Inference",
    long_description=long_description,
    url = "https://github.com/flatironinstitute/inferelator",
    author = "Chris Jackson",
    author_email = "cj59@nyu.edu",
    maintainer = "Chris Jackson",
    maintainer_email = "cj59@nyu.edu",
    packages = ["inferelator"],
    zip_safe = False,
    install_requires = install_requires,
    tests_require = tests_require,
    test_suite = "nose.collector",
)
