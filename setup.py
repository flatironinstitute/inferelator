import os
import sys
from setuptools import setup

install_requires = ["requests"]
#tests_require = ["mock", "unittest2"]
tests_require = ["coverage", "nose"]

base_dir = os.path.dirname(os.path.abspath(__file__))

version = "0.1"

#if sys.argv[-1] == 'publish':
#    os.system("git tag -a %s -m 'v%s'" % (version, version))
#    os.system("python setup.py sdist bdist_wheel upload -r pypi")
#    print("Published version %s, do `git push --tags` to push new tag to remote" % version)
#    sys.exit()

#if sys.argv[-1] == 'syncci':
#    os.system("panci --to=tox .travis.yml > tox.ini");
#    sys.exit();

setup(
    name = "inferelator_ng",
    version = version,
    description = "inferelator next generation",
    long_description="\n\n".join([
        open(os.path.join(base_dir, "README.md"), "r").read(),
        #open(os.path.join(base_dir, "CHANGELOG.rst"), "r").read()
    ]),
    url = "https://github.com/simonsfoundation/inferelator_ng",
    author = "Aaron Watters",
    author_email = "awatters@simonsfoundation.org",
    maintainer = "Aaron Watters",
    maintainer_email = "awatters@simonsfoundation.org",
    packages = ["inferelator_ng"],
    zip_safe = False,
    install_requires = install_requires,
    tests_require = tests_require,
    test_suite = "nose.collector",
)
