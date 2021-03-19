===============
Getting Started
===============
``TransitFit`` operation is based around three :ref:`config files <config files>` (the *data path file*, the *priors file* and the *filter info file*) and a single Python wrapper function (:meth:`~transitfit._pipeline.run_retrieval`). There's a lot more going on under the hood but, for everyday use, you don't need to worry about that.

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

To fit these observations with basic linear detrending and a bog-standard quadratic limb darkening model without using the :ref:`LDC coupling offered by <Limb-darkening>` ``TransitFit``, run::

    from transitfit import run_retrieval

    results = run_retrieval('input_data.csv, priors.csv')

This is a simple approach which is useful for getting some preliminary results, as it is generally the fastest-running method. However, ``TransitFit`` can do better.

Let's now imagine we have decided to use a quadratic detrending model, and also want to take advantage of the :ref:`coupled LDC fitting <Limb-darkening>`. To do this, we still use :meth:`~transitfit._pipeline.run_retrieval`, but have to specify a few more arguments, including providing filter profiles and host information.::

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
^^^^^^^^^^^^^^^^^^^^^^

``TransitFit`` provides a variety of outputs. These are:

Output files
    These are .csv files which record the best fit value and uncertainty for each parameter fitted. The base folder in which they are saved is controlled by the ``results_output_folder`` argument of :meth:`~transitfit._pipeline.run_retrieval`.

    There are a few different versions of these:

    * ``'complete_output.csv'`` - this contains the complete final results of the entire retrieval, collating results from all :ref:`stages of retrieval <Fitting large number of parameters>` if required.

    * ``summary_output_file`` - this contains the best fit results from a specific stage of retrieval (e.g. if in :ref:`'folded mode' <'Folded' fitting>`, this would be the results for a specific filter (stage 1) or the results for fitting the folded curves (stage 2)). The name of this file can be specified with the ``summary_file`` argument of :meth:`~transitfit._pipeline.run_retrieval` and defaults to ``'summary_output.csv'``.

    # ``full_output_file`` - This contains every output from a given batched run, including indication of which batch the results come from. The name of this file can be specified with the ``full_output_file`` argument of :meth:`~transitfit._pipeline.run_retrieval` and defaults to ``'full_output.csv'``.

Fitted light curves
    These are .csv files for each input light curve, containing the normalised and detrended light curves, along with phase values and best-fit models. The columns for these are

    1. **Time** - The time of observation
    2. **Phase** - The phase of observation, setting mid-transit to a phase of 0.5.
    3. **Normalised flux** - The detrended and normalised flux values
    4. **Flux uncertainty** - The uncertainty on the normalised flux
    5. **Best fit curve** - The normalised best-fit light curve.

    By default these are saved in the ``'./fitted_lightcurves'`` folder, but this can be changed using the ``final_lightcurve_folder`` argument of :meth:`~transitfit._pipeline.run_retrieval`.

Plots
    ``TransitFit`` has a few different plots that it provides. The default base folder for these is ``'./plots'`` and this can be set using ``plot_folder='./plots'``. The different plot types are:

    *Fitted light curves*
        These are given for each individual light curve. Additionally, if running in :ref:`'folded mode' <'Folded' fitting>`, a folded curve for each filter is produced. These plots come in versions with and without error bars.

    *Posterior samples*
        These are made using ``corner`` and show the samples for each run of ``dynesty`` within a ``TransitFit`` retrieval.

    *Quick-look folded curves*
        Since running in :ref:`'folded mode' <'Folded' fitting>` can take some time, ``TransitFit`` provides a 'quick-look' plot for each filter after folding. This is there mostly so that you can be satisfied that the folding makes sense, rather than then having to wait until the end to find out something has gone wrong.
