'''
pipeline

A function which will run everything when given a path to light curve and
priors!
'''

from .io import read_data_file_array, read_priors_file, read_input_file, read_filter_info
from .retriever import Retriever
import numpy as np

def run_retrieval_from_paths(input_csv_path, prior_path, filter_info_path=None,
                             host_T=None, host_logg=None, host_z=None,
                             ld_fit_method='single', ld_model='quadratic',
                             n_ld_samples=20000, do_ld_mc=False,
                             allowed_ld_variance=5, detrending=None, nlive=300,
                             normalise=True, low_norm=0.1, high_norm=15):
    '''
    Runs a full retrieval when given data_paths and prior_path

    Parameters
    ----------
    input_csv_path : str
        Path to the file which contains the paths to data files with associated
        epoch and filter numbers. For info on format, see documentation for
        transitfit.io.read_input_file
    prior_path : str
        Path to the prior .csv file. See TransitFit.io.read_priors_file for
        more information on format.
    filter_info_path : str or None, optional
        Path to .csv containing information on the filters. See
        TransitFit.io.read_filter_info for more information on format. If not
        None and host_T, host_logg and host_z are not None, retrieval will
        include fitting realistic limb darkening parameters for the filters.
        Default is None.
    host_T : tuple or None, optional
        The log_10 of the surface gravity of the host star, with gravity
        measured in cm/s2. Should be given as a (value, uncertainty) pair.
        Default is None.
    host_logg : tuple or None, optional
        The log_10 of the surface gravity of the host star, with gravity
        measured in cm/s2. Should be given as a (value, uncertainty) pair.
        Default is None.
    host_z : tuple or None, optional
        The metalicity of the host, given as a (value, uncertainty) pair.
        Default is None.
    ld_fit_method : str, optional
        Determines if limb darkening parameter fitting uses only a single
        waveband, or if all wavebands are considered.
        If 'single' is used, only the first filter is fitted, and the
        rest are extrapolated using the ratios between the 'most likely'
        values given by LDTk for a host with the given parameters.
        If 'all' is used, then the LD parameters for each waveband are all
        treated as independent free parameters and are fitted separately.
        Whilst 'all' is arguably preferrable, it will significantly
        increase the run time when a large number of filters are used.
        Default is 'single'.
    ld_model : str, optional
        The limb darkening model to use. Default is quadratic
    n_ld_samples : int, optional
        The number of samples to take when calculating limb darkening
        coefficients. Default is 20000,
    do_ld_mc : bool, optional
        If True, will use MCMC sampling to more accurately estimate the
        uncertainty on intial limb darkening parameters provided by LDTk.
        Default is False.
    allowed_ld_variance : float, optional
        The number of standard deviations that each parameter is allowed to
        vary by. Default is 5.
    detrending : str or None, optional
        If not None, detrending will be performed. Accepted detrending models
        are 'linear', 'quadratic', 'sinusoidal'. Default is None.
    nlive : int, optional
        The number of live points to use in the nested sampling retrieval.
    normalise : bool, optional
        If True, will assume that the light curves have not been normalised and
        will fit normalisation constants within the retrieval. Default is
        True.
    low_norm : float, optional
        The lowest value to consider as a multiplicative normalisation
        constant. Default is 0.1.
    high_norm : float, optional
        The highest value to consider as a multiplicative normalisation
        constant. Default is 15.

    Returns
    -------
    results : dict
        The results returned by Retriever.run_dynesty()
    '''
    print('Loading light curve data...')
    times, depths, errors = read_input_file(input_csv_path)

    # Read in the priors
    print('Loading priors from {}'.format(prior_path))
    priors = read_priors_file(prior_path, ld_model)

    # Set up all the optional fitting modes (limb darkening, detrending,
    # normalisation...)
    if filter_info_path is not None:
        if host_T is None or host_z is None or host_logg is None:
            raise ValueError('Filter info path was provided but I am missing infomration on the host!')
        print('Loading filter info from {}'.format(filter_info_path))
        filters = read_filter_info(filter_info_path)

        print('Initialising limb darkening fitting...')
        priors.fit_limb_darkening(host_T, host_logg, host_z, filters, ld_model,
                                  ld_fit_method, n_ld_samples, do_ld_mc,
                                  allowed_ld_variance)

    if detrending is not None:
        print('Initialising detrending')
        priors.add_detrending(times, detrending)

    if normalise:
        print('Initialising normalisation...')
        priors.fit_normalisation(depths, default_low=low_norm)

    print('Beginning retrieval of {} parameters'.format(len(priors.fitting_params)))
    retriever = Retriever()
    return retriever.run_dynesty(times, depths, errors, priors, nlive=nlive)
