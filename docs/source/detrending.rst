==========
Detrending
==========

``TransitFit`` has the capability to detrend light curves simultaneously with fitting physical parameters, and can handle using both nth-order polynomial and user-specified detrending functions. It is able to fit multiple detrending models for different observations at once, which is particularly useful when combining observations from different telescopes which have known systematic properties.

For nth-order detrending, we assume that the detrending is additive, and that the detrended flux, :math:`\mathbf{D}(\mathbf{t})`, is given by

.. math::
    \mathbf{D}(\mathbf{t}) = \mathbf{F}(\mathbf{t}) - \mathbf{d}(\mathbf{t})

where :math:`\mathbf{F}(\mathbf{t})` is the raw flux and :math:`\mathbf{d}(\mathbf{t})` is the detrending function. However, if you have a more complicated detrending function which has multiplicative elements, or value which depend on the actual flux, ``TransitFit`` can use a custom model to do this.

We will look here at how to get ``TransitFit`` to use the different types of detrending, and show a simple example of setting up a custom detrending function.

Basic detrending syntax
^^^^^^^^^^^^^^^^^^^^^^^

Setting up ``TransitFit`` to use different detrending models is simple and uses the ``detrending_list`` kwarg in :meth:`~transitfit._pipeline.run_retrieval`. This is a list of the different detrending methods to be used, along with any required details. The :ref:`detrending indices  <Indexing>` given in the :ref:`data input file <Data input file>` refer to the index of the methods in this list.

Detrending methods
------------------

The available types of detrending are:

*Nth order*
    To use a polynomial of order ``n``, the entry to ``detrending_list`` should be given as ``['nth order', n]``. The nth order polynomials used by ``TransitFit`` are designed to be flux-conserving, and are of the form

    .. math::
        d\left(t_i\right) = \sum^n_{j=1} \left[a_j \left(t_i^j - \overline{\textbf{t}^j}\right)\right]


    The full derivation of this can be found in the `paper <https://ui.adsabs.harvard.edu/abs/2021arXiv210312139H>`_

*Custom function*
    Using a custom function requires a little more information. By default, all parameters are assumed to be global: that is, there is a single value for each parameter which applies to *all light curves with this detrending model*. There are situations where some parameters in a detrending function should not be fitted globally. We define three cases of this:

    * **"telescope dependent" parameters** - ones where a single parameter value applies to *all light curves observed with the same telescope*.

    * **"wavelength dependent"** parameters - ones where a single parameter value applies to *all light curves observed at the same wavelength*

    * **"epoch dependent"** parameters - ones where a single parameter value applies to *all light curves observed at the same time*.

    Custom detrending functions must take a :meth:`~transitfit.LightCurve` as their first argument, and each argument after that must be a float. It must return the detrended flux values. Aside from this, there are no major restrictions to the type of detrending you can use.

    Let's assume that we want to use the following arbitrary detrending function::

        def f(lightcurve, a, b, c):

            times = lightcurve.times

            detrending_vals = times - a * exp(-b * times) + c
            detrended_flux = lightcurve.flux - detrending_vals
            return

    and that ``c`` is some wavelength dependent parameter.

    The general syntax to use for a custom detrending function ``f()`` is::

        ['custom', f, [[telescope dependent parameters], [wavelength dependent parameters], [epoch dependent parameters]]]

    To specify that a parameter is telescope-, wavelength-, or epoch-dependent, add the index of the relevant argument to the appropriate list. In our example, our entry for ``c`` being wavelength dependent would be::

        ['custom', f, [[], [3], []]]


*No detrending*
    To not detrend a light curve, use ``['off']`` in your ``detrending_list``


Setting limits on detrending coefficients
-----------------------------------------

By default, all detrending coefficients are fitted using a uniform prior of :math:`\pm10`. Obviously this is not always ideal, so you can specify the range over which these priors should be fitted using the ``detrending_limits`` argument in :meth:`~transitfit._pipeline.run_retrieval`. **Note**: all the detrending coefficients in a given model will be bound to the same range.

To use custom ranges in your detrending models, you use a list where each entry is ``[lower, upper]`` for the detrending methods.


An example
----------

Let's again consider our :ref:`toy model <Some toy observations>` with three observations. We shall assume that we want to apply a quadratic detrending model to one, the custom detrending model above to another, and that the last one has already been detrended in pre-processing. We will also change the coefficient bounds. We first need to edit our ``'input_data.csv'`` to::

    Path,                   Telescope,  Filter,     Epoch,      Detrending
    /path/to/observation1,  0,          0,          0,          0
    /path/to/observation2,  0,          1,          1,          1
    /path/to/observation3,  1,          0,          2,          2

and then our full input code, using the coupled LDC fitting, becomes::

    from transitfit import run_retrieval

    # Set up the custom detrending function
    def f(times, a, b, c):
        return times - a * exp(-b * times) + c

    # Set up the host info, using arbitrary values.
    # These are all given in (value, uncertainty) tuples
    host_T = (5450, 130) # Effective temperature in Kelvin
    host_z = (0.32, 0.09) # The metalicity
    host_r = (1.03, 0.05) # Host radius in solar radii - this MUST be supplied if the prior for orbital separation is in AU.
    host_logg = (4.5, 0.1) # log10(suface gravity) in cm/s2

    # Set up the detrending models
    detrending_models = [['nth order', 2],  # This is detrending index 0
                         ['custom', f, [[3], [], []]],  # This is detrending index 1
                         ['off']]  # This is detrending index 2

    # Set the detrending coefficient bounds
    detrending_limits = [[-10, 10],  # bounds for model 0
                         [-3, 20],  # bounds for model 1
                         [0.2, 4.8]]  # bounds for model 2

    # Now we can run the retrieval!
    results = run_retrieval('input_data.csv', 'priors.csv', 'filter_profiles.csv',  # Config paths
                            detrending_list=detrending_models,  # Set up detrending models
                            detrending_limits=detrending_limits  # Set the detrending limits
                            ld_fit_method='coupled'  # Turn on coupled LDC fitting
                            host_T=host_T, host_logg=host_logg, host_z=host_z, host_r=host_r  # host params)
