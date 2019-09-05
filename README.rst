================
TransitFit
================

**Author** - Joshua Hayes (joshjchayes@gmail.com)

TransitFit is a package designed to use nested sampling to run retrieval on exoplanet transit light curves. The transit model used is batman

Contents
========
`Requirements`_

`Setting Priors`_

`Running Retrieval`_



Requirements
============
Along with an installation of Python 3 (with the standard Conda distribution packages), TransitFit requires the following packages to be installed:

- dynesty_

- batman_


Setting Priors
==============
Before running retrieval, the priors must be set up. This requires the creation of a ``PriorInfo`` object. There are two ways to do this.

Method 1: create ``PriorInfo`` and add fitting parameters
-------------------------------------------------------------
The first method involves creating a ``PriorInfo`` object and setting default value for all parameters, and then adding bounds to parameters you want to fit. To create a :code:`PriorInfo` object, you can use::

  prior_info = transitfit.setup_priors(args)

The docstring of ``setup_priors`` explains the full usage of this and the arguments. Once you have the ``PriorInfo`` set up, you can add a parameter to be fitted over using::

  prior_info.add_uniform_fit_param(name, best, low_lim, high_lim, light_curve_num=None)

The complete list of parameters and the names recognised by TransitFit are:

- ``'P'``: Period of the orbit

- ``'rp'``: Planet radius (in stellar radii)

- ``'t0'``: time of inferior conjunction

- ``a'`` : semi-major axis (in units of stellar radii)

- ``'inc'``: inclination of the orbit

- ``'ecc'``: eccentricity of the orbit

- ``'w'``: longitude of periastron (in degrees)



Method 2: create ``PriorInfo`` from a .csv file
-----------------------------------------------
If you don't want to mess around with directly coding everything, TransitFit can read in your priors from a .csv file! To do this, use::

  prior_info = transitfit.read_priors_file('/path/to/file/')

The docstring will tell you more about how this should be presented.


Running Retrieval
=================
To run retrieval, set up a ``Retriever`` object::

  retriever = transitfit.Retriever()

You also need to have your light curves. These need to be laid out as 3 (M,N) arrays, one for each of the time series, flux from the star (normalised), and the uncertainty on the flux. In the shape, M is the number of light curves you want to run retrieval on simultaneously. Note that these must all be for the same planet!!

Now we can use the ``PriorInfo`` and ``Retriever`` together to get some results::

  results = retriever.run_dynesty(times, depths, errors, prior_info)



.. _dynesty: https://dynesty.readthedocs.io/en/latest/index.html
.. _batman: https://www.cfa.harvard.edu/~lkreidberg/batman/index.html
