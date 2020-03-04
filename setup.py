'''
setup script for transitfit

'''

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transitfit",
    version="0.9.1",
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
                      'pandas', 'ldtk']
)
