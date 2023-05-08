import os
from setuptools import setup, find_packages

# Current Inferelator Version Number
version = "0.6.2"


# Description from README.md
base_dir = os.path.dirname(os.path.abspath(__file__))
long_description = "\n\n".join([open(os.path.join(base_dir, "README.md"), "r").read()])

# Required packages
requires = [
    "numpy >=1.14.0",
    "scipy >=0.9",
    "pandas >=0.24.0",
    "scikit-learn",
    "matplotlib >=1.5.1",
    "anndata >=0.9.0 "
]

setup(
    name="inferelator",
    version=version,
    description="Inferelator: Network Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flatironinstitute/inferelator",
    project_urls={
        "Documentation": "https://inferelator.readthedocs.io/",
    },
    author="Chris Jackson",
    author_email="cj59@nyu.edu",
    maintainer="Chris Jackson",
    maintainer_email="cj59@nyu.edu",
    packages=find_packages(include=["inferelator", "inferelator.*"], exclude=["tests", "*.tests"]),
    zip_safe=False,
    install_requires=requires,
    python_requires=">=3.5",
    tests_require=["coverage", "pytest", "bio-test-artifacts", "tables"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta"
    ]
)
