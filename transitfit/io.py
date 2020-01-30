'''
file_reader

Module to help make using csv files super easy as inputs
'''

import numpy as np
import pandas as pd
from .priorinfo import PriorInfo, _prior_info_defaults, setup_priors
import dynesty
import csv
import os
from ._utils import validate_variable_key

def parse_filter_list(filter_list):
    '''
    Parses a list of filter information into a usable form for the `'filters'`
    argument in PriorInfo.fit_limb_darkening.

    Parameters
    ----------
    filter_list : array_like, shape (n_filters, 3)
        The information on the filters. Each row should contain
        [filter_index, low wavelength, high wavelength]
        The filter indices should refer to the indices used in the priors file.
        Wavelengths should be in nm.

    Returns
    -------
    filter_info : np.array, shape (n_filters, 2)
        The filter information pass to PriorInfo.fit_limb_darkening.
    '''
    filter_list = np.array(filter_list)

    # How many filters are there?
    n_filters = int(filter_list[:,0].max() + 1)

    # Make a blank array to populate with the filter limits
    filter_info = np.zeros((n_filters, 2))

    # Now populate!
    for i in range(n_filters):
        filter_info[i] = filter_list[i, 1:]

    return filter_info


def parse_data_path_list(data_path_list):
    '''
    Parses a list of paths to data files and places them into an array which
    can be passed to read_data_file_array

    Parameters
    ----------
    data_path_list : array_like, shape (n_light_curves, 3)
        The list of paths. Each row should contain
        [data path, epoch index, filter index]

    Returns
    -------
    data_path_array : array_like, shape (num_filters, num_epochs)
        The paths inserted into an array where each column is a particular
        filter and each row is a particular epoch of observation.
    '''
    data_path_list = np.array(data_path_list, dtype=object)

    if data_path_list.ndim == 1:
        data_path_list = np.array([data_path_list])

    # Work out how many epochs and filters we have
    n_epochs = data_path_list[:,1].max() + 1
    n_filters = data_path_list[:,2].max() + 1

    # Initialise a blank array, filling it with None.
    data_path_array = np.array([[None for i in range(n_epochs)] for j in range(n_filters)], object)

    # Populate the blank array
    for row in data_path_list:
        p, i, j = row
        data_path_array[j,i] = p

    return data_path_array


def parse_priors_list(priors_list, ld_model, num_filters, num_epochs):
    '''
    Parses a list of priors to produce a PriorInfo with all fitting parameters
    initialised.

    Parameters
    ----------
    priors_list : array_like, shape(X, 5)
        A list of prior information for each variable to be fitted. Can also
        set fixed values by setting the low and high values to `None` or
        `np.nan`. Each row should be of the form
        [key, default value, low limit, high limit, filter index]
    ld_model : str
        The limb darkening model to use
    num_filters : int
        The number of different filters being used
    num_epochs : int
        The number of different epochs being used.

    Returns
    -------
    prior_info : PriorInfo
        The fully initialised PriorInfo which can then be used in fitting.
    '''
    priors_dict = {}

    for row in priors_list:
        # First check the key and correct if possible
        row[0] = validate_variable_key(row[0])

        # Now add to the priors_dict
        priors_dict[row[0]] = row[1:]

    ##############################
    # Make the default PriorInfo #
    ##############################

    # Need to check if any variables are missing from the default prior
    for key in _prior_info_defaults:
        if key not in priors_dict:
            # Has not been specified in the priors list
            priors_dict[key] = [_prior_info_defaults[key]]

    # Now make the basic PriorInfo
    prior_info = setup_priors(priors_dict['P'][0],
                              priors_dict['rp'][0],
                              priors_dict['a'][0],
                              priors_dict['inc'][0],
                              priors_dict['t0'][0],
                              priors_dict['ecc'][0],
                              priors_dict['w'][0],
                              ld_model, num_epochs, num_filters)

    ##########################
    # Initialise the fitting #
    ##########################

    for row in priors_list:
        key, best, low, high, filt = row
        # Check this has been given as a value to fit and not just specified
        if low is None:
            low = np.nan
        if high is None:
            high = np.nan
        if np.isfinite(low) and np.isfinite(high):
            if key in ['rp']:
                prior_info.add_uniform_fit_param(key, best, low, high, filter_idx=int(filt))
            else:
                prior_info.add_uniform_fit_param(key, best, low, high)

    return prior_info


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


def read_data_file(path, skiprows=0, delimiter=None, folder=None):
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
    folder : str or None, optional
        If not None, this folder will be prepended to all the paths. Default is
        None.

    Returns
    -------
    times : np.array
        The times of the data series
    flux : np.array
        The flux
    error : np.array
        The uncertainty on the flux
    '''
    if folder is None:
        folder = ''

    if path[-4:] == '.csv':
        return _read_data_csv(os.path.join(folder, path))
    if path[-4:] == '.txt':
        return _read_data_txt(os.path.join(folder, path), skiprows)

def read_data_path_array(data_path_array, skiprows=0):
    '''
    If passed an array of paths, will read in to produce times, flux and
    uncertainty arrays
    '''
    data_path_array = np.array(data_path_array)

    num_wavelengths = data_path_array.shape[0]
    num_times = data_path_array.shape[1]

    data = np.array([[None for i in range(num_times)] for j in range(num_wavelengths)], object)

    for i in range(num_wavelengths):
        for j in range(num_times):
            if data_path_array[i,j] is not None:
                data[i,j] = read_data_file(data_path_array[i,j])


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

def read_priors_file(path, num_epochs, num_filters, limb_dark='quadratic'):
    '''
    If given a csv file containing priors, will produce a PriorInfo object
    based off the given values

    Parameters
    ----------
    path : str
        Path to .csv file containing the priors.

        Columns should me in the order
        --------------------------------------------------------
        |  key  |   best  |  low_lim  |   high_lim  |  filter  |
        --------------------------------------------------------

        If the parameter is invariant across an epoch or filter, leave the
        entry blank.

        If you want to fix a parameter at a given value, leave low_lim and
        high_lim blank. Just provide best, along with epoch and filter if
        required
    num_epochs : int
        The number of observation epochs which exist in the data
    num_filters : int
        The number of different filters which are used in the observations

    limb_dark : str, optional
        The model of limb darkening you want to use. Accepted are
            - linear
            - quadratic
            - squareroot
            - power2
            - nonlinear
        Default is quadratic

    Notes
    -----
    Detrending currently cannot be initialised in the prior file. It will be
    available as a kwarg in the pipeline function
    '''
    priors_list = pd.read_csv(path).values

    return parse_priors_list(priors_list, limb_dark, num_filters, num_epochs)


def read_input_file(path):
    '''
    Reads in a file with listed inputs and produces data arrays for use in
    retrieval.

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

    data_path_array = parse_data_path_list(info)

    return read_data_path_array(data_path_array)

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

    return parse_filter_list(info)


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

    #resampled = dynesty.utils.resample_equal(results.samples, results.weights)

    # Put the output into a dictionary that's nice to deal with
    out_dict = {}
    write_dict = []

    # TODO: add in single/all modes for limb darkening parameters
    if priorinfo.ld_fit_method == 'single':
        # We've only fitted one wavelengths' LD coeffs:
        # Estimate the rest of the limb darkening values and write.

        # Get the fitted values in a usable format
        best_single_ld = np.zeros(len(priorinfo.limb_dark_coeffs))
        for u, ui in enumerate(priorinfo.limb_dark_coeffs):
            x = np.where(priorinfo.fitting_params == u)[0]
            best_single_ld[u] = best[x]

        # Now do the estimation
        best_ld_params = priorinfo.ld_param_handler.estimate_values(best_single_ld)
        # TODO errors


    for i, param in enumerate(priorinfo.fitting_params):
        if param in priorinfo.limb_dark_coeffs:
            # We need to convert the LDCs
            LDC_index = int(param[-1])
            LDC_unit_vals = best[i - LDC_index : i + len(priorinfo.limb_dark_coeffs) - LDC_index]
            LDC_unit_uncs = results.uncertainties[i - LDC_index : i + len(priorinfo.limb_dark_coeffs) - LDC_index]
            value = round(priorinfo.ld_handler.convert_coefficients(*LDC_unit_vals)[LDC_index], 6)
            unc = round(priorinfo.ld_handler.convert_coefficients(*LDC_unit_uncs)[LDC_index], 6)

        else:
            value = best[i]
            unc = results.uncertainties[i]

        if param in ['rp']:
            param = param +'_{}'.format(int(priorinfo._filter_idx[i]))
        #elif param in ['t0']:
        #    param = param +'_{}'.format(int(priorinfo._epoch_idx[i]))
        elif param in priorinfo.detrending_coeffs + ['norm']:
            param = param + '_f{}_e{}'.format(int(priorinfo._filter_idx[i]), int(priorinfo._epoch_idx[i]))
        elif param in priorinfo.limb_dark_coeffs and priorinfo.ld_fit_method in ['independent', 'coupled']:
            # All the LD coeffs are fitted separately and will write out
            param = param +'_{}'.format(int(priorinfo._filter_idx[i]))


        out_dict[param] = value

        write_dict.append({'Parameter': param, 'Best value':value,
                           'Uncertainty' : unc})


    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Now output to a .csv
    with open(filepath, 'w') as f:
        #columns = ['Parameter', 'Best value', 'Lower error', 'Upper error', 'Median']
        columns = ['Parameter', 'Best value', 'Uncertainty']
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(write_dict)


def print_results(results, priorinfo, n_dof):
    '''
    Prints the results nicely to terminal

    Parameters
    ----------
    results : dynesty.results.Results
        The Dynesty results object, but must also have weights, cov and
        uncertainties as entries.
    priorinfo : transitfit.priorinfo.PriorInfo
        The PriorInfo object
    '''
    best = results.samples[np.argmax(results.logl)]

    print('\nBest fit results:')

    # We need to print out the results. Loop over each fitted
    for i, param in enumerate(priorinfo.fitting_params):

        if param in priorinfo.limb_dark_coeffs:
            # We need to convert the LDCs
            LDC_index = int(param[-1])
            LDC_unit_vals = best[i - LDC_index : i + len(priorinfo.limb_dark_coeffs) - LDC_index]
            LDC_unit_uncs = results.uncertainties[i - LDC_index : i + len(priorinfo.limb_dark_coeffs) - LDC_index]
            value = round(priorinfo.ld_handler.convert_coefficients(*LDC_unit_vals)[LDC_index], 6)
            unc = round(priorinfo.ld_handler.convert_coefficients(*LDC_unit_uncs)[LDC_index], 6)

        else:
            value = round(best[i], 6)
            unc = round(results.uncertainties[i], 6)

        if param in ['rp']:
            param = param +'_{}:\t'.format(int(priorinfo._filter_idx[i]))
        #elif param in ['t0']:
        #    param = param +'_{}'.format(int(priorinfo._epoch_idx[i]))
        elif param in priorinfo.detrending_coeffs + ['norm']:
            param = param + '_f{}e{}:'.format(int(priorinfo._filter_idx[i]), int(priorinfo._epoch_idx[i]))
        elif param in priorinfo.limb_dark_coeffs and priorinfo.ld_fit_method in ['independent', 'coupled']:
            # All the LD coeffs are fitted separately and will write out
            param = param +'_{}:\t'.format(int(priorinfo._filter_idx[i]))
        else:
            param += ':\t'
        print('{}\t {} ± {}'.format(param, value, unc))

    best_chi2 = - results.logl.max()

    print('chi2:\t\t {}'.format(round(best_chi2, 5)))
    print('red chi2:\t {}'.format(round(best_chi2/n_dof, 5)))


def save_final_light_curves(times, flux, uncertainty, priorinfo, results,
                            folder='./final_light_curves'):
    '''
    Applies detrending and normalisation to each light curve and saves to .csv

    Parameters
    ----------
    results : dynesty.results.Results
        The Dynesty results object, but must also have weights, cov and
        uncertainties as entries.
    priorinfo : transitfit.priorinfo.PriorInfo
        The PriorInfo object
    folder : str, optional
        The folder to save the files to. Default is './final_light_curves'
    '''
    # Get some numbers for loop purposes
    n_epochs = priorinfo.num_times
    n_filters = priorinfo.num_wavelengths

    # Get the best values
    best = results.samples[np.argmax(results.logl)]
    best_dict = priorinfo._interpret_param_array(best)

    os.makedirs(folder, exist_ok=True)

    # Loop over each light curve and apply detrending and normalisation
    for fi in range(n_filters):
        for ei in range(n_epochs):
            if times[fi, ei] is not None:

                norm = best_dict['norm'][fi, ei]

                if priorinfo.detrend:
                    d = [best_dict[d][fi, ei] for d in priorinfo.detrending_coeffs]

                    dF = priorinfo.detrending_function(times[fi, ei]-np.floor(times[fi, ei][0]), *d)

                    detrended_flux = norm * (flux[fi, ei] - dF)
                else:
                    # No detrending, just normalisation
                    detrended_flux = norm * flux[fi, ei]

                normalised_uncertainty = norm * uncertainty[fi, ei]


                write_dict = []
                for i, ti in enumerate(times[fi, ei]):
                    #print(i)
                    write_dict.append({'Time' : ti,
                                       'Normalised flux' : detrended_flux[i],
                                       'Uncertainty' : normalised_uncertainty[i]})

                fname = 'f{}_e{}_detrended.csv'.format(fi, ei)
                with open(os.path.join(folder, fname), 'w') as f:
                    columns = ['Time', 'Normalised flux', 'Uncertainty']
                    writer = csv.DictWriter(f, columns)
                    writer.writeheader()
                    writer.writerows(write_dict)
