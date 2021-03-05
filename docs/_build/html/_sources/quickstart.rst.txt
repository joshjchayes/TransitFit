===============
Getting Started
===============
``TransitFit`` operation is based around three :ref:`config files <config files>` (the *data path file*, the *priors file* and the *filter info file*) and a single Python wrapper function (:meth:`~transitfit.run_retrieval`). There's a lot more going on under the hood but, for everyday use, you don't need to worry about that.

A note on time standards
^^^^^^^^^^^^^^^^^^^^^^^^
**TransitFit assumes that all time values are in BJD**. If you are using BJD-2450000, or any variation on this, you will be okay as long as you are consistent. Using HJD, local time, or any time system that is not based on BJD will end in incorrect results and probably tears.

An illustration
^^^^^^^^^^^^^^^

Some toy observations
---------------------

Let's assume we have observations of 3 transits of a planet with a 6-day period:

*Observation 1*
    Taken with telescope A, using an R-band filter, on Day 0.

*Observation 2*
    Taken with telescope A, using a filter with uniform transmission between 400nm and 500nm, on Day 12.

*Observation 3*
    Taken with telescope B, using an R-band filter, on Day 42.

The config files for this might look something like

* ``'input_data.csv'``::

    Path,                   Telescope,  Filter,     Epoch,      Detrending
    /path/to/observation1,  0,          0,          0,          0
    /path/to/observation2,  0,          1,          1,          0
    /path/to/observation3,  1,          0,          2,          0

* ``'priors.csv'``::

    Parameter, Distribution, InputA,        InputB, Filter
    P,         gaussian,     6,             0.001,
    t0,        gaussian,     2457843.45246, 0.007,
    a,         gaussian,     7.64,          0.5,
    inc,       gaussian,     88.5,          1.2,
    rp,        uniform,      0.13,          0.19,   0
    rp,        uniform,      0.13,          0,19,   1
    ecc,       fixed,        0,             ,

* ``'filter_profiles.csv'``::

    Filter index,   InputA,   InputB
    0,              R,
    1,              400,      500

See :ref:`here <Config files>` for more information on exactly what these mean and how to set them up.

Running ``TransitFit`` on these data
------------------------------------
Once we have the config files set up, it is incredibly easy to retrieve a best-fit model using ``TransitFit``.

To fit these observations with basic linear detrending and a bog-standard quadratic limb darkening model without using the LDC coupling offered by ``TransitFit``, run::

    from transitfit import run_retrieval

    results = run_retrieval('input_data.csv, priors.csv')

This is a simple approach which is useful for getting some preliminary results, as it is generally the fastest-running method. However, ``TransitFit`` can do better.

Let's now imagine we have decided to use a quadratic detrending model, and also want to take advantage of the coupled LDC fitting. To do this, we still use :meth:`~transitfit.run_retrieval`, but have to specify a few more arguments, including providing filter profiles and host information.::

    from transitfit import run_retrieval

    # Set up the host info, using arbitrary values.
    # These are all given in (value, uncertainty) tuples
    host_T = (5450, 130) # Effective temperature in Kelvin
    host_z = (0.32, 0.09) # The metalicity
    host_r = (1.03, 0.05) # Host radius in solar radii - this MUST be supplied if the prior for orbital separation is in AU.
    host_logg = (4.5, 0.1) # log10(suface gravity) in cm/s2

    # Set up the detrending model
    # We want to use a quadratic (2nd order) model
    detrending_models = [['nth order', 2]]

    # Now we can run the retrieval!
    results = run_retrieval('input_data.csv', 'priors.csv', 'filter_profiles.csv',  # Config paths
                            detrending_list=detrending_models,  # Set up detrending models
                            ld_fit_method='coupled'  # Turn on coupled LDC fitting
                            host_T=host_T, host_logg=host_logg, host_z=host_z, host_r=host_r  # host params)


``TransitFit`` outputs
----------------------
Talk about the things!!
