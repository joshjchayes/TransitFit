Welcome to TransitFit!
======================

``TransitFit`` is a Python 3.X package for fitting multi-telescope, multi-filter, and multi-epoch exoplanetary transit observations. It uses the `batman <https://www.cfa.harvard.edu/~lkreidberg/batman/index.html>`_ transit model and nested sampling routines from `dynesty <https://dynesty.readthedocs.io/en/latest/index.html>`_.

``TransitFit`` is unique among publicly available transit-fitting codes as its likelihood calculations can include the effect of host star characteristics and observation filter profiles on limb-darkening coefficients (LDCs), which we refer to as :ref:`'coupling' the LDCs <Limb-darkening>`. It can also perform per-telescope detrending simultaneously with the fitting of other parameters.

See :ref:`Getting Started` for instructions on how to use ``TransitFit``. You can also find more information in the `TransitFit paper <https://ui.adsabs.harvard.edu/abs/2021arXiv210312139H/abstract>`_


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

    pip install transitfit

Alternatively, you can install it direct from the source by downloading the project from the `GitHub page <https://github.com/joshjchayes/TransitFit>`_ and running::

    python setup.py install


Citing
======
If you find ``TransitFit`` useful in your work, please cite the `accompanying paper <https://ui.adsabs.harvard.edu/abs/2021arXiv210312139H>`_. If you are using BibTeX, you can use the following citation in your .bib file::

    @ARTICLE{2021arXiv210312139H,
           author = {{Hayes}, J.~J.~C. and {Kerins}, E. and {Morgan}, J.~S. and {Humpage}, A. and {Awiphan}, S. and {Charles-Mindoza}, S. and {McDonald}, I. and {Inyanya}, T. and {Padjaroen}, T. and {Munsaket}, P. and {Chuanraksasat}, P. and {Komonjinda}, S. and {Kittara}, P. and {Dhillon}, V.~S. and {Marsh}, T.~R. and {Reichart}, D.~E. and {Poshyachinda}, S.},
            title = "{TransitFit: an exoplanet transit fitting package for multi-telescope datasets and its application to WASP-127~b, WASP-91~b, and WASP-126~b}",
          journal = {arXiv e-prints},
         keywords = {Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
             year = 2021,
            month = mar,
              eid = {arXiv:2103.12139},
            pages = {arXiv:2103.12139},
    archivePrefix = {arXiv},
           eprint = {2103.12139},
     primaryClass = {astro-ph.EP},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210312139H},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


.. toctree::
    :hidden:
    :maxdepth: 2

    quickstart
    configfiles
    limb_darkening
    detrending
    manyparams
    ttvs
    faqs
    api
