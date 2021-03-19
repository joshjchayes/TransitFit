==================================
Fitting large number of parameters
==================================

In an ideal world, a transmission spectrum would be made up of very many light curve observations, at many filters and epochs. Fitting all of these observations simultaneously would result in a very high-dimensioned parameter space, which leads to instabilities in the nested sampling.

In order to avoid this, ``TransitFit`` has modes for dealing with large numbers of parameter sets: ``'batched'``, and ``'folded'``, which can be set with the ``fitting_mode`` argument of :meth:`~transitfit._pipeline.run_retrieval`. There is also ``'all'`` mode, which can be set to manually force all parameters to be fitted simultaneously.

'Batched' fitting
-----------------

This mode groups the light curves by filter, and then splits the retrieval into multi-filter 'batches,' fitting each of these batches one at a time. The batches are chosen to allow a maximum number of parameters to be fitted simultaneously, which can be controlled with the ``max_batch_parameters`` argument of :meth:`~transitfit._pipeline.run_retrieval`. Final best-fit values are then calculated from weighted means of the best-fit values from each batch.

The batches are generally constructed to have at least one filter in common with at least one other batch. This ensures that there is still some coupling of information between the batches. The exception to this is when there is one filter in particular which has a very high number of observations. In this case, we recommend using the ``'folded'`` mode.


'Folded' fitting
----------------
With the launch of large surveys such as *TESS*, many exoplanets have multiple-epoch observations in a single filter. ``TransitFit`` can make use of these through a two-step retrieval process.

    1. ``TransitFit`` runs a retrieval on each filter independently, and uses the results to produce a phase-folded light curve for each filter.

    2. ``TransitFit`` runs a standard multi-wavelength retrieval using the batched algorithm above.

This mode of retrieval allows for the production of high-quality folded light curves from non-detrended data, as well as providing a method where observations from long-term, single-waveband surveys such as *TESS* can be easily combined with single-epoch observations at multiple wavelengths, such as from ground-based spectrographic follow-up.


Automatic mode selection
------------------------

By default, ``TransitFit`` will try and detect which fitting mode is most appropriate for your data. It first works out how many parameters are required to fit everything simultaneously, which we will call ``max_n_params``. Then if:

    1. ``max_n_params <= max_batch_parameters`` - all parameters are fitted simultaneously (``'all'`` mode).

    2. If any filter has 3 or more epochs observed, then ``'folded'`` mode is used.

    3. Otherwise, ``'batched'`` mode is used.
