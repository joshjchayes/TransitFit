Welcome to TransitFit!
======================

``TransitFit`` is a Python 3.X package for fitting multi-filter, multi-epoch exoplanetary transit observations.

``TransitFit`` is unique among publicly available transit-fitting codes as its likelihood calculations can include the effect of host star characteristics and observation filter profiles on limb-darkening coefficients (LDCs), which we refer to as :ref:`'coupling' the LDCs <Limb-darkening>`.

See :ref:`Getting Started` for instructions on how to use ``TransitFit``.


Installation
============
``TransitFit`` is compatible with Python 3.X installations. To run, ``TransitFit`` requires the following packages:

* numpy
* scipy
* pandas
* matplotlib
* corner
* `batman <https://www.cfa.harvard.edu/~lkreidberg/batman/index.html>`_
* `dynesty <https://dynesty.readthedocs.io/en/latest/index.html>`_
* `ldtk <https://github.com/hpparvi/ldtk>`_

To install the most recent stable version, run::

    CODE TO FOLLOW.

Alternatively, you can install it direct from the source by downloading the project from the GitHub page and running::

    python setup.py install


Citing
======
If you find ``TransitFit`` useful in your work, please cite the `LINK TO PAPER <DEAD LINK>`_. If you are using BibTeX, you can use the following citation in your .bib file::

    BIB things to go here


.. toctree::
    :hidden:
    :maxdepth: 2

    quickstart
    configfiles
    limb_darkening
    detrending
    api
