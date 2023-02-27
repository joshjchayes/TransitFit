============
Config Files
============
There are three configuration files required by ``TransitFit``. These are the *data input file*, the *priors file* and the *filter info file*. These are all fairly prescriptive in their format, and should be provided as .csv files with one header row.

For the purposes of illustration, we will be using an example where we have observations of 3 transits of a planet with a 6-day period:

*Observation 1*
    Taken with telescope A, using an R-band filter, on Day 0.

*Observation 2*
    Taken with telescope A, using a filter with uniform transmission between 400nm and 500nm, on Day 12.

*Observation 3*
    Taken with telescope B, using an R-band filter, on Day 42.

We will also assume that we are going to only use one :ref:`detrending model <Detrending>`.

Indexing
--------
In order to allow for full flexibility, ``TransitFit`` operates on a zero-indexed indexing system. Each light curve is identified by three indices: **telescope**, **filter**, and **epoch**. The actual ordering of these indices does not matter, as long as they are consistent across all light curves being fitted, and no values are skipped. A useful practice is to order observation epochs chronologically (0 being the first observation) and filters in either ascending or descending order of wavelength.

This approach makes it possible to easily fit (for example) two simultaneous observations with the same filter from two different sites, or single wavelength, multi-epoch observations from a single observatory, or any other combination you can think of.

Since ``TransitFit`` offers the ability to use multiple different detrending models simultaneously, a **detrending method** index is also required. This is particularly useful in situations where space- and ground-based observations are being combined, since space-based observations often require something more complex than an nth-order approach. This is discussed in more depth :ref:`here <Detrending>`.

So, to summarise, each transit observation in the dataset that you are fitting is identified by a **telescope**, **filter**, and **epoch** index and the detrending model is controlled by a **detrending method** index.



Data input file
---------------
This file is used to direct ``TransitFit`` to the light curve observations that you want to fit. It is also where the :ref:`indexing scheme <Indexing>` for the light curves is defined. Each transit observation to be fitted should be a separate text (.txt or .csv) file, with three columns: time (in BJD), flux, and uncertainty on flux. Note that the light curves do not have to be normalised or detrended, as this is :ref:`something that TransitFit can handle <Detrending>`. The data input file should have 5 columns:

1. **Path**: this is the absolute or relative path to the data file for each light curve.
2. **Telescope index**: The index associated with the telescope used to take this observation.
3. **Filter index**: The index which identifies the filter used in this observation.
4. **Epoch index**: The index which identifies the epoch of this transit. Note that this does not reflect the number of transits that have passed. For example, if you have two observations of a planet with a 5 day orbit, taken a year apart, the indices would be 0 and 1, **not** 0 and 72.
5. **Detrending method index**: This is the index used to choose which detrending method you want to use. See :ref:`here <Detrending>` for more info.

In the case of our example above, the ``input_data.csv`` file will look something like::

    Path,                   Telescope,  Filter,     Epoch,      Detrending
    /path/to/observation1,  0,          0,          0,          0
    /path/to/observation2,  0,          1,          1,          0
    /path/to/observation3,  1,          0,          2,          0


Priors
------
This file determines which physical parameters are to be fitted by ``TransitFit``, and the distribution from which samples are to be drawn from for each. It can also be used to manually fix a parameter to a specific value. This file should also have 5 columns:

1. **Parameter**: The parameters which can be set using the priors file are

    * ``P``: Period of the orbit, in BJD
    * ``t0``: time of inferior conjunction in BJD
    * ``a`` : semi-major axis. This can be given in units of either host-radii or AU. If given in AU, then ``host_r`` must be specified in :meth:`~transitfit._pipeline.run_retrieval` to allow for a conversion to host-radii.
    * ``inc``: inclination of the orbit in degrees (Defaults to 90 degrees if not provided)
    * ``ecc``: eccentricity of the orbit (defaults to 0 if not provided)
    * ``w``: longitude of periastron (in degrees) (Defaults to 90 degrees if not provided)
    * ``rp``: Planet radius in stellar radii (i.e. Rp/R\*). **Note**: if you have multiple filters that you want to fit ``rp`` for, you will have to provide a prior for *each* filter.
    * {``q0``, ``q1``, ``q2``, ``q3``} : Kipping q parameters for limb darkening. Most of the time you will not need to set these, but if you want to run a retrieval without fitting for limb darkening (if, for example, you fitted for these another way), then you can set them here by specifying a ``'fixed'`` distribution. Note that you will also have to set ``ld_fit_method='off'`` in the arguments of :meth:`~transitfit._pipeline.run_retrieval`.

2. **Distribution**: The distribution that samples will be drawn from. This can be any of:

    * ``uniform`` - uses a uniform, box-shaped prior
    * ``gaussian`` - uses a Gaussian prior
    * ``fixed`` - the parameter won't be fitted and will be fixed at a user-specified value.

3. **Input A**: The use of this column depends on the distribution being used:

    * If ``uniform``: provide the **lower bound** of the uniform distribution.
    * If ``gaussian``: provide the **mean** of the Gaussian distribution.
    * If ``fixed``: provide the value to fix the parameter at.

4. **Input B**: The use of this column depends on the distribution being used:

    * If ``uniform``: provide the **upper bound** of the uniform distribution.
    * If ``gaussian``: provide the **standard deviation** of the Gaussian distribution.
    * If ``fixed``: this input is not used and anything here will be ignored.

5. **Filter index**: If a parameter varies with wavelength (i.e. ``rp`` and limb-darkening coefficients), the filter index must be supplied for each instance in the priors file, making sure to follow the indexing set out in the data paths and filter info files.

So, for our example observations, if we assume a circular orbit (i.e. don't fit for ``ecc`` and ``w``), our ``'priors.csv'`` file might look something like::

    Parameter, Distribution, InputA,        InputB, Filter
    P,         gaussian,     6,             0.001,
    t0,        gaussian,     2457843.45246, 0.007,
    a,         gaussian,     7.64,          0.5,
    inc,       gaussian,     88.5,          1.2,
    rp,        uniform,      0.13,          0.19,   0
    rp,        uniform,      0.13,          0,19,   1
    ecc,       fixed,        0,             ,

When setting up your priors, we recommend that you use a uniform distribution for ``rp`` so that you don't inadvertently bias the values, especially if you're doing spectroscopy work.

Filter profiles
----------------
This file is used to specify the filter profiles that observations were made at, and is only required if you are using ``TransitFit``'s :ref:`ability to couple LDCs across wavelengths <Limb-darkening>`.

TransitFit can deal with either uniform box filters (useful for narrow-band spectroscopy), or full filter response functions. It comes pre-packaged with a set of standard filters:

* Johnson-Cousins *UVRIB*
* SLOAN-SDSS *u'g'r'i'z'*
* The *TESS* filter
* The *Kepler* filter

If you want to use your own filter profile, you can provide a .csv with 2 columns: wavelength in nm, and filter transmission, either as a fraction or percentage (``TransitFit`` will detect which).

The filter info file requires 3 columns:

1. **Filter index**: the index of the filter, ensuring consistency with the other input files.

2. **Input A**:

    * For a uniform box filter, provide the **lowest wavelength** not blocked by the filter in nanometres.
    * The name of one of the provided filter profiles: any of: U, V, R, I, B, u', g', r', i', z', TESS, Kepler.
    * The path to a user-provided filter profile

3. **Input B**:
    * For a uniform box filter, provide the **highest wavelength** not blocked by the filter in nanometres.
    * For anything else, this column is ignored.

So, for our example, ``filter_profiles.csv`` would look like::

    Filter index,   InputA,   InputB
    0,              R,
    1,              400,      500
