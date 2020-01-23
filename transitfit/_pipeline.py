'''
pipeline

A function which will run everything when given a path to light curve and
priors!
'''

from .io import read_priors_file, read_input_file, read_filter_info, parse_data_path_list, read_data_path_array, parse_priors_list, parse_filter_list
from .retriever import Retriever
import numpy as np



def run_retrieval(data_files, priors, ld_model='quadratic',
                  ld_fit_method='independent', filter_info=None, host_T=None,
                  host_logg=None, host_z=None, ldc_low_lim=-5, ldc_high_lim=-5,
                  n_ld_samples=20000, do_ld_mc=False, detrending='nth order',
                  detrending_order=1, detrending_function=None, nlive=300,
                  normalise=True, low_norm=0.1, dlogz=None):
    '''
    Runs a full retrieval of posteriors using nested sampling on a transit
    light curve or a set of transit light curves.

    Parameters
    ----------
    data_files : str or array_like, shape (n_light_curves, 3)
        Either a path to a file which contains the paths to data files with
        associated epoch and filter numbers, or a list of paths.
        If a path to a .csv file is provided, columns should be in the order
        -------------------------------
        |  path  |  epoch  |  filter  |
        -------------------------------
        If a list in provided, each row should contain
        [data path, epoch index, filter index]
    priors : str or array_like, shape(X, 5)
        Path to a .csv file containing prior information on each variable to be
        fitted, or a list containing the same information. The columns for
        either should be in the order
        ---------------------------------------------------------------
        | key | default value | low limit | high limit | filter index |
        ---------------------------------------------------------------
        If you want to fix a parameter at a given value, leave low_lim and
        high_lim blank (if in a file) or set to `None` or `np.nan`. Just
        provide best, along with epoch and filter if required.

        The accepted keys are:
            - 'P' : orbital period. Should be in the same time units as t0
            - 'rp' : planet radius (filter specific). One should be provided
                     for each filter being used
            - 'a' : orbital radius in host radii
            - 'inc' : orbital inclination in degrees
            - 't0' : a t0 value for one of the light curves being used. This
                     should be in the same time units as the period
            - 'ecc' : the orbital eccentricity
            - 'w' : the longitude of periastron (in degrees)
    ld_model : str, optional
        The limb darkening model to use. Allowed models are
            - 'linear'
            - 'quadratic'
            - 'squareroot'
            - 'power2'
            - 'nonlinear'
        With the exception of the non-linear model, all models are constrained
        by the method in Kipping (2013), which can be found at
        https://arxiv.org/abs/1308.0009. Default is 'quadratic'.
    ld_fit_method : str, optional
        Determines the mode of fitting of limb darkening parameters. The
        available modes are:
            - `'coupled'` : all limb darkening parameters are fitted
              independently, but are coupled to a wavelength dependent
              model based on the host parameters through `ldkt`
            - `'single'` : LD parameters are still tied to a model, but only
              the first filter is actively fitted. The remaining filters
              are estimated based off the ratios given by ldtk for a host
              with the given parameters. This mode is useful for a large
              number of filters, as 'coupled' or 'independent' fitting will
              lead to much higher computation times.
            - `'independent'` : Each LD coefficient is fitted separately for
              each filter, with no coupling to the ldtk models.
        Default is `'independent'`
    filter_info : str or array_like, shape (n_filters, 3) or None, optional
        Path to a .csv file containing prior information on each variable to be
        fitted, or a list containing the same information. The columns for
        either should be in the order
        -----------------------------------------
        |   filter_idx  |   low_wl  |   high_wl |
        -----------------------------------------
        This is required if ld_fit_method is `'single'` or `'coupled'`. If not
        None and host_T, host_logg and host_z are not None, retrieval will
        include fitting realistic limb darkening parameters for the filters.
        Default is None.
    host_T : tuple or None, optional
        The effective temperature of the host star, in Kelvin. Should be given
        as a (value, uncertainty) pair. Required if ld_fit_method is `'single'`
        or `'coupled'`. Default is None.
    host_logg : tuple or None, optional
        The log_10 of the surface gravity of the host star, with gravity
        measured in cm/s2. Should be given as a (value, uncertainty) pair.
        Required if ld_fit_method is `'single'` or `'coupled'`. Default is None
    host_z : tuple or None, optional
        The metalicity of the host, given as a (value, uncertainty) pair.
        Required if ld_fit_method is `'single'` or `'coupled'`. Default is None
    ldc_low_lim : float, optional
        The lower limit to use in conversion in the case where there are
        open bounds on a limb darkening coefficient (power2 and nonlinear
        models). Note that in order to conserve sampling density in all regions
        for the power2 model, you should set lower_lim=-high_lim. Default is -5
    ldc_high_lim : float, optional
        The upper limit to use in conversion in the case where there are
        open bounds on a limb darkening coefficient (power2 and nonlinear
        models). Note that in order to conserve sampling density in all regions
        for the power2 model, you should set lower_lim=-high_lim. Default is 5
    n_ld_samples : int, optional
        Controls the number of samples taken by PyLDTk when calculating LDCs
        when using 'coupled' or 'single' modes for limb darkening fitting.
        Default is 20000
    do_ld_mc : bool, optional
        If True, will use MCMC sampling to more accurately estimate the
        uncertainty on intial limb darkening parameters provided by PyLDTk.
        Default is False.
    detrending : str or None, optional
        If not None, detrending will be performed. Accepted detrending models
        are
            - 'nth order'
            - 'custom'.
        'nth order' must have a detrending_order value supplied and will
        detrend whilst obeying a flux conservation law. If 'custom', then
        detrending_function must be provided. Default is 'nth order'.
    detrending_order : int, optional
        The order of detrending function to apply if detrending is 'nth order'.
        The default is 1, corresponding to linear detrending.
    detrending_funcs : None or function, optional
        The detrending function. If provided and detrending is 'custom', will
        apply this as the detrending function. We assume that the first
        argument is times, and that all others are single valued -
        TransitFit cannot fit list/array variables. Default is None
    nlive : int, optional
        The number of live points to use in the nested sampling retrieval.
    normalise : bool, optional
        If True, will assume that the light curves have not been normalised and
        will fit normalisation constants within the retrieval. The range to
        fit normalisation constants c_n are automatically detected using
            ``1/f_median - 1 <= c_n <= 1/f_median + 1``
        as the default range, where f_median is the median flux value for a
        given light curve. ``low_norm`` can be used to adjust the default
        minimum value in the case that ``1/f_median - 1  < 0``. Default is
        True.
    low_norm : float, optional
        The lowest value to consider as a multiplicative normalisation
        constant. Default is 0.1.
    dlogz : float, optional
        Retrieval iteration will stop when the estimated contribution of the
        remaining prior volume to the total evidence falls below this
        threshold. Explicitly, the stopping criterion is
        `ln(z + z_est) - ln(z) < dlogz`, where z is the current evidence
        from all saved samples and z_est is the estimated contribution from
        the remaining volume. The default is `1e-3 * (nlive - 1) + 0.01`.

    Returns
    -------
    results : dict
        The results returned by Retriever.run_dynesty()
    '''
    print('Loading light curve data...')

    if type(data_files) == str:
        times, depths, errors = read_input_file(data_files)
    else:
        data_path_array = parse_data_path_list(data_files)
        times, depths, errors = read_data_path_array(data_path_array)

    num_epochs = times.shape[1]
    num_filters = times.shape[0]

    # Read in the priors
    if type(priors) == str:
        print('Loading priors from {}...'.format(priors))
        priors = read_priors_file(priors, num_epochs, num_filters, ld_model)
    else:
        print('Initialising priors...')
        priors = parse_priors_list(priors, ld_model, num_filters, num_epochs)

    # Set up all the optional fitting modes (limb darkening, detrending,
    # normalisation...)
    if ld_fit_method == 'independent':
        print('Initialising limb darkening fitting...')
        priors.fit_limb_darkening(ld_fit_method, ldc_low_lim, ldc_high_lim, ld_model=ld_model)

    elif ld_fit_method in ['coupled', 'single']:
        if filter_info is None:
            raise ValueError('filter_info must be provided for coupled and single ld_fit_methods')
        if host_T is None or host_z is None or host_logg is None:
            raise ValueError('Filter info was provided but I am missing information on the host!')

        if type(filter_info) == str:
            print('Loading filter info from {}...'.format(filter_info))
            filters = read_filter_info(filter_info)
        else:
            print('Initialising filter infomation...')
            filters = parse_filter_list(filter_info)

        print('Initialising limb darkening fitting...')
        priors.fit_limb_darkening(ld_fit_method, ldc_low_lim, ldc_high_lim, host_T,
                                  host_logg, host_z, filters, ld_model,
                                  n_ld_samples, do_ld_mc)



    if detrending is not None:
        print('Initialising detrending')
        priors.add_detrending(times, detrending, order=detrending_order, function=detrending_function)

    if normalise:
        print('Initialising normalisation...')
        priors.fit_normalisation(depths, default_low=low_norm)

    print('The parameters we are retrieving are: {}'.format(priors.fitting_params))
    print('Beginning retrieval of {} parameters'.format(len(priors.fitting_params)))
    retriever = Retriever()
    return retriever.run_dynesty(times, depths, errors, priors, nlive=nlive, dlogz=dlogz)
