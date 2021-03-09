'''
setup script for transitfit

'''

import setuptools
import os
import codecs

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transitfit",
    version=get_version('transitfit/__init__.py'),
    author="Joshua Hayes",
    author_email="joshjchayes@gmail.com",
    description="A package to fit exoplanet transit light curves",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joshjchayes/TransitFit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta"
    ],
    python_requires='>=3.6',
    install_requires=['batman-package', 'dynesty', 'numpy', 'matplotlib',
                      'pandas', 'ldtk', 'corner', 'semantic_version', 'scipy',],
    zip_safe=False,
    package_data={'transitfit' : ['../filters/*.csv']}
)
