'''
file_reader

Module to help make using csv files super easy as inputs
'''

import numpy as np
import pandas as pd
from .priorinfo import PriorInfo, _prior_info_defaults
import dynesty
import csv

def _read_data_csv(path):
    '''
    Given a path to a csv with columns [time, depths, errors], will get
    all the data in a way which can be used by the Retriever

    '''
    # Read in with pandas
    data = pd.read_csv(path)

    # Extract the arrays
    times, depths, errors = data.values.T

    return times, depths, errors

def _read_data_txt(path, skiprows=0):
    '''
    Reads a txt data file with columns
    '''
    times, depth, errors = np.loadtxt(path, skiprows=skiprows).T

    return times, depths, errors

def read_data_file(path, skiprows=0, delimiter=None):
    '''
    Reads a file in, assuming that it is either a:
        .csv
        .txt
    with columns in the order time, depth, errors. Note that TransitFit assumes
    BJD times.

    Parameters
    ----------
    path : str
        Full path to the file to be loaded
    skiprows : int, optional
        Number of rows to skip in reading txt file (to avoid headers)
    delimiter : str, optional
        The string used to separate values. The default is whitespace.

    Returns
    -------
    times : np.array
        The times of the data series
    flux : np.array
        The flux
    error : np.array
        The uncertainty on the flux
    '''
    if path[-4:] == '.csv':
        return _read_data_csv(path)
    if path[-4:] == '.txt':
        return _read_data_txt(path, skiprows)

def read_data_file_array(data_paths, skiprows=0):
    '''
    If passed an array of paths, will read in to produce times, flux and
    uncertainty arrays
    '''
    data_paths = np.array(data_paths)

    num_wavelengths = data_paths.shape[0]
    num_times = data_paths.shape[1]

    data = np.array([[None for i in range(num_times)] for j in range(num_wavelengths)], object)

    for i in range(num_wavelengths):
        for j in range(num_times):
            if data_paths[i,j] is not None:
                data[i,j] = read_data_file(data_paths[i,j])


    times = np.array([[None for i in range(num_times)] for j in range(num_wavelengths)], object)
    depths = np.array([[None for i in range(num_times)] for j in range(num_wavelengths)], object)
    errors = np.array([[None for i in range(num_times)] for j in range(num_wavelengths)], object)

    for i in range(num_wavelengths):
        for j in range(num_times):
            if data[i,j] is not None:
                times[i,j] = data[i,j][0]
                depths[i,j] = data[i,j][1]
                errors[i,j] = data[i,j][2]

    return times, depths, errors

def read_priors_file(path, limb_dark='quadratic'):
    '''
    If given a csv file containing priors, will produce a PriorInfo object
    based off the given values

    Parameters
    ----------
    path : str
        Path to .csv file containing the priors.

        Columns should me in the order
        ------------------------------------------------------------------
        |  key  |   best  |  low_lim  |   high_lim  |  epoch  |  filter  |
        ------------------------------------------------------------------

        If the parameter is invariant across an epoch or filter, leave the
        entry blank.

        If you want to fix a parameter at a given value, leave low_lim and
        high_lim blank. Just provide best, along with epoch and filter if
        required

    limb_dark : str, optional
        The model of limb darkening you want to use. Accepted are
            - uniform
            - linear
            - quadratic
            - logarithmic
            - exponential
            - squareroot
            - power2
            - nonlinear
        Default is quadratic

    Notes
    -----
    Detrending currently cannot be initialised in the prior file. It will be
    available as a kwarg in the pipeline function
    '''
    table = pd.read_csv(path).values

    # Work out how mnay light curves are being used.
    num_times = sum(table[:, 0] == 't0')
    num_wavelengths = sum(table[:, 0] == 'rp')

    # check the limb darkening coefficients and if they are fitting
    # First, A small dict to check if each LD param is being fitted.
    # This basically checks which parameters are required for the different models
    limb_dark_params = {}
    limb_dark_params['u1'] = False
    limb_dark_params['u2'] = False
    limb_dark_params['u3'] = False
    limb_dark_params['u4'] = False

    if not limb_dark == 'uniform':
        limb_dark_params['u1'] = np.any(table[0] == 'u1')
        if not limb_dark == 'linear':
            limb_dark_params['u2'] = np.any(table[0] == 'u2')
            if limb_dark == 'nonlinear':
                limb_dark_params['u3'] = np.any(table[0] == 'u3')
                limb_dark_params['u4'] = np.any(table[0] == 'u4')

    #######################################################
    # Set up the default dict to initialise the PriorInfo #
    #######################################################
    default_prior_dict = {}

    default_prior_dict['num_times'] = num_times
    default_prior_dict['num_wavelengths'] = num_wavelengths

    # Initialse any variables which vary with epoch or wavelength
    default_prior_dict['rp'] = np.full(num_wavelengths, np.nan)
    default_prior_dict['t0'] = np.full(num_times, np.nan)

    for key in limb_dark_params:
        if limb_dark_params[key]:
            default_prior_dict[key] = np.full(num_wavelengths, np.nan)

    # Now make the default values for the fitting parameters
    for row in table:
        key, best, low, high, epoch, filt = row
        if key == 'rp':
            default_prior_dict[key][int(filt)] = best
        elif key == 't0':
            default_prior_dict[key][int(epoch)] = best
        elif key in ['u1','u2','u3','u4']:
            if limb_dark_params[key]:
                default_prior_dict[key][int(filt)] = best
        else:
            default_prior_dict[key] = best

    # Now we set the fixed values from defaults
    for key in _prior_info_defaults:
        if key not in default_prior_dict:
            default_prior_dict[key] = _prior_info_defaults[key]

    # Check to see if there are any light curves which have not had 'rp' or 't0 defined'
    if np.isnan(default_prior_dict['rp']).any():
        bad_indices = np.where(np.isnan(default_prior_dict['rp']))[0]
        bad_string = str(bad_indices)[1:-1]
        raise ValueError('Light curve(s) {} are missing rp values'.format(bad_string))
    if np.isnan(default_prior_dict['t0']).any():
        bad_indices = np.where(np.isnan(default_prior_dict['t0']))[0]
        bad_string = str(bad_indices)[1:-1]
        raise ValueError('Light curve(s) {} are missing t0 values'.format(bad_string))
    for key in limb_dark_params:
        if limb_dark_params[key]:
            if np.isnan(default_prior_dict[key]).any():
                bad_indices = np.where(np.isnan(default_prior_dict[key]))[0]
                bad_string = str(bad_indices)[1:-1]
                raise ValueError('Light curve(s) {} are missing {}} values'.format(bad_string, key))

    # MAKE DEFAULT PriorInfo #
    prior_info = PriorInfo(default_prior_dict, warn=False)

    #################################
    # Now add in the actual priors! #
    #################################
    for row in table:
        key, best, low, high, epoch, filt = row

        # This is a check to make sure we want to vary the parameter
        if np.isfinite(low) and np.isfinite(high):
            if key in ['rp']:
                prior_info.add_uniform_fit_param(key, best, low, high, filter_idx=int(filt))

            elif key in ['t0']:
                prior_info.add_uniform_fit_param(key, best, low, high, epoch_idx=int(epoch))

            elif key in ['u1','u2','u3','u4']:
                prior_info.add_uniform_fit_param(key, best, low, high, filter_idx=int(filt))

            else:
                prior_info.add_uniform_fit_param(key, best, low, high)

    return prior_info

def read_input_file(path):
    '''
    Reads in a file with listed inputs and produces data arrays for use in
    retrieval. This can be used to

    Parameters
    ----------
    path : str
        The path to the .csv file with the paths to input parameters and
        their filter and epoch number.

        Columns should be in the order
        -------------------------------
        |  path  |  epoch  |  filter  |
        -------------------------------

    Returns
    -------
    times : np.array
        The times of each data point
    flux : np.array
        The fluxes
    uncertainty : np.array
        The uncertainty on each flux
    '''
    info = pd.read_csv(path).values

    # Work out how many epochs and filters we have
    n_epochs = info[:,1].max() + 1
    n_filters = info[:,2].max() + 1

    # Initialise a blank array
    paths_array = np.array([[None for i in range(n_epochs)] for j in range(n_filters)], object)

    for row in info:
        p, i, j = row
        paths_array[j,i] = p

    return read_data_file_array(paths_array)

def read_filter_info(path):
    '''
    Reads in information on the filters from .csv file and puts them in a
    format which can be passed to the ``filters`` argument in
    PriorInfo.fit_limb_darkening.

    Parameters
    ----------
    path : str
        Path to the .csv file containing the information on the filters. This
        file should have three columns:

        -----------------------------------------
        |   filter_idx  |   low_wl  |   high_wl |
        -----------------------------------------

        The filter indices should refer to the indices used in the priors file.
        All wavelengths should be in nm.

    Returns
    -------
    filter_info : np.array, shape (n_filters, 2)
        The filter information pass to PriorInfo.fit_limb_darkening.

    '''

    info = pd.read_csv(path).values

    # How many filters are there?
    n_filters = int(info[:,0].max() + 1)

    # Make a blank array to populate with the filter limits
    filter_info = np.zeros((n_filters, 2))

    # Now populate!
    for i in range(n_filters):
        filter_info[i] = info[i, 1:]

    return filter_info

def save_results(results, priorinfo, filepath='outputs.csv'):
    '''
    Saves results from a retrieval to csv file

    Parameters
    ----------
    results : dict
        A results dictionary as returned by Retriever.run_dynesty()
    priorinfo : PriorInfo
        The PriorInfo that the run used
    filepath : str, optional
        The path to save the output to. Default is a file called outputs.csv in
        the current directory.
    '''

    if not filepath[-4:] =='.csv':
        filepath += '.csv'

    best = results.samples[np.argmax(results.logl)]

    resampled = dynesty.utils.resample_equal(results.samples, results.weights)

    # Put the output into a dictionary that's nice to deal with
    out_dict = {}
    write_dict = []

    # TODO: add in single/all modes for limb darkening parameters

    for i, param in enumerate(priorinfo.fitting_params):
        value = best[i]
        lower = np.percentile(resampled[:,i], 16)
        upper = np.percentile(resampled[:,i], 84)
        median = np.median(resampled[:, i])

        if param in ['rp'] + priorinfo.limb_dark_coeffs:
            param = param +'_{}'.format(int(priorinfo._filter_idx[i]))
        elif param in ['t0']:
            param = param +'_{}'.format(int(priorinfo._epoch_idx[i]))
        elif param in priorinfo.detrending_coeffs:
            param = param + '_f{}_e{}'.format(int(priorinfo._filter_idx[i]), int(priorinfo._epoch_idx[i]))
        out_dict[param] = value

        write_dict.append({'Parameter': param, 'Best value':value,
                           'Lower error' : median - lower ,
                           'Upper error' : upper - median,
                           'Median' : median})

    # Now output to a .csv
    with open(filepath, 'w') as f:
        columns = ['Parameter', 'Best value', 'Lower error', 'Upper error', 'Median']
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(write_dict)
