'''
pipeline

A function which will run everything when given a path to light curve and
priors!
'''

from .io import read_priors_file, read_input_file, read_filter_info, parse_data_path_list, read_data_path_array, parse_priors_list, parse_filter_list
from .retriever import Retriever
import numpy as np



def run_retrieval(data_files, priors, detrending_list=[['nth order', 1]],
                  ld_model='quadratic', ld_fit_method='independent',
                  filter_info=None, data_skiprows=0, host_T=None,
                  host_logg=None, host_z=None, ldc_low_lim=-5, ldc_high_lim=-5,
                  n_ld_samples=20000, do_ld_mc=False, nlive=300,
                  normalise=True, low_norm=0.1, dlogz=None, maxiter=None,
                  maxcall=None,  output_param_path='./outputs.csv',
                  final_lightcurve_folder='./final_light_curves',
                  plot_folder='./plots', plot_best=True, figsize=(12,8),
                  plot_color='dimgrey', plot_titles=None, add_plot_titles=True,
                  plot_fnames=None, cache_path=None):
    '''
    Runs a full retrieval of posteriors using nested sampling on a transit
    light curve or a set of transit light curves.

    Parameters
    ----------
    data_files : str or array_like, shape (n_light_curves, 5)
        Either a path to a file which contains the paths to data files with
        associated epoch and filter numbers, or a list of paths.
        If a path to a .csv file is provided, columns should be in the order
        ------------------------------------------------------------
        |  path  |  telescope  |  filter  |  epoch  |  detrending  |
        ------------------------------------------------------------
        where detrending is an index which refers to the detrending method from
        detrending_method_list to use for a given light curve
        If a list in provided, each row should contain
        [data path, telescope idx, filter idx, epoch idx, detrending idx]
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
            - 'w' : the longitude of periastron in degrees
    detrending_list : array_like, shape (n_detrending_models, 2)
        A list of different detrending models. Each entry should consist
        of a method and a second parameter dependent on the method.
        Accepted methods are
            ['nth order', order]
            ['custom', function]
            ['off', ]
        function here is a custom detrending function. TransitFit assumes
        that the first argument to this function is times and that all
        other arguments are single-valued - TransitFit cannot fit
        list/array variables. If 'off' is used, no detrending will be
        applied to the `LightCurve`s using this model. The default is
        [['nth order', 1]], giving linear detrending.
    ld_model : str, optional
        The limb darkening model to use. Allowed models are
            - 'linear'
            - 'quadratic'
            - 'squareroot'
            - 'power2'
            - 'nonlinear'
        With the exception of the non-linear model, all models are constrained
        by the method in Kipping (2013), which can be found at
        https://arxiv.org/abs/1308.0009. Use `ldc_low_lim` and `ldc_high_lim`
        to control the behaviour of unconstrained coefficients.
        Default is 'quadratic'.
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
    output_param_path : str, optional
        Path to save output csv to. Default is './outputs.csv'
    final_lightcurve_folder : str, optional
        The folder to save normalised and detrended light curves to. Default
        is './final_light_curves'
    plot_folder : str, optional
        Path to folder to save plots to. Default is './plots'
    plot_best : bool, optional
        If True, will plot the data and the best fit model on a Figure.
        Default is True.
    figsize : tuple, optional
        The fig size for each plotted figure. Default is (12, 8)
    plot_color : matplotlib color, optional
        The base color for plots. Default is 'dimgray'
    plot_titles : None or array_like, shape (n_filters, n_epochs), optional
        The titles to use for each plot. If None, will default to
        'Filter X Epoch Y'. Default is None.
    add_plot_titles : bool, optional
        If True, will add titles to plots. Default is True.
    plot_fnames : None or array_like, shape (n_filters, n_epochs), optional
        The file names to use for each plot. If None, will default to
        'fX_eY.pdf'. Default is None.
    cache_path : str, optional
        This is the path to cache LDTK files to. If not specified, will
        default to the LDTK default

    Returns
    -------
    results : dict
        The results returned by Retriever.run_dynesty()
    '''
    print('Loading light curve data...')

    if type(data_files) == str:
        lightcurves, detrending_index_array = read_input_file(data_files)
    else:
        data_path_array, detrending_index_array = parse_data_path_list(data_files)
        lightcurves = read_data_path_array(data_path_array)

    n_telescopes = lightcurves.shape[0]
    n_filters = lightcurves.shape[1]
    n_epochs = lightcurves.shape[2]

    # Read in the priors
    if type(priors) == str:
        print('Loading priors from {}...'.format(priors))
        priors = read_priors_file(priors, n_telescopes, n_filters, n_epochs, ld_model)
    else:
        print('Initialising priors...')
        priors = parse_priors_list(priors, n_telescopes, n_filters, n_epochs, ld_model)

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
                                  n_ld_samples, do_ld_mc, cache_path)

    print('Initialising detrending')
    priors.add_detrending(lightcurves, detrending_list, detrending_index_array)

    if normalise:
        print('Initialising normalisation...')
        priors.fit_normalisation(lightcurves, default_low=low_norm)

    print('The parameters we are retrieving are: {}'.format(priors.fitting_params))
    print('Beginning retrieval of {} parameters'.format(len(priors.fitting_params)))
    retriever = Retriever()
    return retriever.run_dynesty(lightcurves, priors,
                                 maxiter=maxiter, maxcall=maxcall, nlive=nlive,
                                 dlogz=dlogz, savefname=output_param_path,
                                 plot=plot_best, plot_folder=plot_folder,
                                 figsize=figsize, plot_color=plot_color,
                                 output_folder=final_lightcurve_folder,
                                 plot_titles=plot_titles,
                                 add_plot_titles=add_plot_titles,
                                 plot_fnames=plot_fnames)
