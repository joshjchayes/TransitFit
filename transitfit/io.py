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
from .lightcurve import LightCurve
import batman
import os

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
    data_path_list : array_like, shape (n_light_curves, 5)
        The list of paths. Each row should contain
        [data path, telescope idx, filter idx, epoch idx, detrending idx]

    Returns
    -------
    data_path_array : array_like, shape (num_filters, num_epochs, num_telescopes)
        The paths inserted into an array which can be used by TransitFit
    detrending_index_array : array_like, shape (num_filters, num_epochs, num_telescopes)
        The array of detrending model indices for each light curve
    '''
    data_path_list = np.array(data_path_list, dtype=object)

    if data_path_list.ndim == 1:
        data_path_list = np.array([data_path_list])

    # Work out how many epochs and filters we have
    n_telescopes = data_path_list[:,1].max() + 1
    n_filters = data_path_list[:,2].max() + 1
    n_epochs = data_path_list[:,3].max() + 1

    n_detrending_models = data_path_list[:,4].max() + 1

    # Initialise a blank array, filling it with None.
    data_path_array = np.full((n_telescopes, n_filters, n_epochs), None, object)
    detrending_index_array = np.full((n_telescopes, n_filters, n_epochs), None, object)


    # Populate the blank array
    for row in data_path_list:
        p, i, j, k, l = row
        data_path_array[i, j, k] = p
        detrending_index_array[i, j, k] = l

    return data_path_array, detrending_index_array


def parse_priors_list(priors_list, n_telescopes, n_filters,
                     n_epochs, ld_model):
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
    n_telescopes : int
        The number of different telescopes being used. Required so that
        simultaneous observations from different observatories can be used by
        TransitFit
    n_filters : int
        The number of different filters being used
    n_epochs : int
        The number of different epochs being used
    ld_model : str
        The limb darkening model to use

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
                              ld_model, n_telescopes, n_filters, n_epochs)

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
    Given a path to a csv with columns [time, flux, errors], will get
    all the data in a way which can be used by the Retriever

    '''
    # Read in with pandas
    data = pd.read_csv(path)

    # Extract the arrays
    times, flux, errors = data.values.T

    return times, flux, errors

def _read_data_txt(path, skiprows=0):
    '''
    Reads a txt data file with columns
    '''
    times, depth, errors = np.loadtxt(path, skiprows=skiprows).T

    return times, flux, errors


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
    If passed an array of paths, will read in to produce an array of
    `LightCurve`s

    Parameters
    ----------
    data_path_array : array_like, shape (n_telescopes, n_filters, n_epochs)
        Array with paths to data to load

    Returns
    -------
    lightcurves  : np.array of `LightCurve`s, shape (n_telescopes, n_filters, n_epochs)
        The data loaded and stored as an array of LightCurves
    '''
    data_path_array = np.array(data_path_array)

    n_telescopes = data_path_array.shape[0]
    n_filters = data_path_array.shape[1]
    n_epochs = data_path_array.shape[2]

    lightcurves = np.full((data_path_array.shape), None, object)

    for i in np.ndindex(lightcurves.shape):
        if data_path_array[i] is not None:
            times, flux, errors = read_data_file(data_path_array[i])

            lightcurves[i] = LightCurve(times, flux, errors, i[0], i[1], i[2])

    return lightcurves

def read_priors_file(path, n_telescopes, n_filters, n_epochs,
                     limb_dark='quadratic'):
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

        If the parameter is invariant across a filter, leave the
        entry blank.

        If you want to fix a parameter at a given value, leave low_lim and
        high_lim blank. Just provide best, along with filter if required
    n_telescopes : int
        The number of different telescopes being used. Required so that
        simultaneous observations from different observatories can be used by
        TransitFit
    n_filters : int
        The number of different filters being used
    n_epochs : int
        The number of different epochs being used
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

    return parse_priors_list(priors_list, n_telescopes, n_filters, n_epochs, limb_dark)


def read_input_file(path, skiprows=0):
    '''
    Reads in a file with listed inputs and produces data arrays for use in
    retrieval.

    Parameters
    ----------
    path : str
        The path to the .csv file with the paths to input parameters and
        their telescope, filter, epoch, and detrending indices.

        Columns should be in the order
        ------------------------------------------------------------
        |  path  |  telescope  |  filter  |  epoch  |  detrending  |
        ------------------------------------------------------------

    Returns
    -------
    lightcurves  : np.array of `LightCurve`s, shape (n_telescopes, n_filters, n_epochs)
        The data loaded and stored as an array of LightCurves
    detrending_index_array : array_like, shape (n_telescopes, n_filters, n_epochs)
        The detrending indices for each lightcurve
    '''
    info = pd.read_csv(path).values

    data_path_array, detrending_index_array = parse_data_path_list(info)

    lightcurves = read_data_path_array(data_path_array, skiprows=skiprows)

    return lightcurves, detrending_index_array

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

    for i, param in enumerate(priorinfo.fitting_params):
        if not (param in priorinfo.limb_dark_coeffs and priorinfo.ld_fit_method == 'single'):
            if param in priorinfo.limb_dark_coeffs:
                LDC_index = int(param[-1])
                LDC_vals = best[i - LDC_index : i + len(priorinfo.limb_dark_coeffs) - LDC_index]
                LDC_uncs = results.uncertainties[i - LDC_index : i + len(priorinfo.limb_dark_coeffs) - LDC_index]
                value = LDC_vals[LDC_index]
                unc = LDC_uncs[LDC_index]

            else:
                value = best[i]
                unc = results.uncertainties[i]

            if param in ['rp']:
                param = param +'_{}'.format(int(priorinfo._filter_idx[i]))
            #elif param in ['t0']:
            #    param = param +'_{}'.format(int(priorinfo._epoch_idx[i]))
            elif param in priorinfo.detrending_coeffs + ['norm']:
                param = param + '_t{}_f{}_e{}'.format(int(priorinfo._telescope_idx[i]),int(priorinfo._filter_idx[i]), int(priorinfo._epoch_idx[i]))
            elif param in priorinfo.limb_dark_coeffs and priorinfo.ld_fit_method in ['independent', 'coupled']:
                # All the LD coeffs are fitted separately and will write out
                param = param +'_{}'.format(int(priorinfo._filter_idx[i]))

            out_dict[param] = value

            write_dict.append({'Parameter': param, 'Best value':value,
                               'Uncertainty' : unc})

    # Deal with 'single' fitting LD param mode
    if priorinfo.ld_fit_method == 'single':
        # We've only fitted one wavelengths' LD coeffs:
        # Estimate the rest of the limb darkening values and write.

        # Get the fitted values in a usable format
        best_single_ld = np.zeros(len(priorinfo.limb_dark_coeffs))
        single_uncertainty = np.zeros(len(priorinfo.limb_dark_coeffs))

        for i, ui in enumerate(priorinfo.limb_dark_coeffs):
            x = np.where(np.array(priorinfo.fitting_params) == ui)[0]
            best_single_ld[i] = best[x]
            single_uncertainty = results.uncertainties[x]

        # Now do the estimation
        best_ld_params = priorinfo.ld_handler.ldtk_estimate(best_single_ld)

        # We estimate the errors by using the ratios
        estim_errors = priorinfo.ld_handler.ldtk_estimate(single_uncertainty)

        # Loop through each filter and write out the values
        for i, fi in enumerate(best_ld_params):
            for j, qj in enumerate(priorinfo.limb_dark_coeffs):
                param = qj + '_{}'.format(i)
                value = fi[j]
                unc = estim_errors[i, j]

                write_dict.append({'Parameter': param, 'Best value':value,
                                   'Uncertainty' : unc})
        # Now that we have the values, we can write out.

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
        if not (param in priorinfo.limb_dark_coeffs and priorinfo.ld_fit_method == 'single'):
            if param in priorinfo.limb_dark_coeffs:
                # We need to convert the LDCs
                LDC_index = int(param[-1])
                LDC_vals = best[i - LDC_index : i + len(priorinfo.limb_dark_coeffs) - LDC_index]
                LDC_uncs = results.uncertainties[i - LDC_index : i + len(priorinfo.limb_dark_coeffs) - LDC_index]
                value = LDC_vals[LDC_index]
                unc = LDC_uncs[LDC_index]

            else:
                value = round(best[i], 6)
                unc = round(results.uncertainties[i], 6)

            if param in ['rp']:
                param = param +'_{}:\t'.format(int(priorinfo._filter_idx[i]))
            #elif param in ['t0']:
            #    param = param +'_{}'.format(int(priorinfo._epoch_idx[i]))
            elif param in priorinfo.detrending_coeffs + ['norm']:
                param = param + '_t{}_f{}_e{}:'.format(int(priorinfo._telescope_idx[i]),int(priorinfo._filter_idx[i]), int(priorinfo._epoch_idx[i]))
            elif param in priorinfo.limb_dark_coeffs and priorinfo.ld_fit_method in ['independent', 'coupled']:
                # All the LD coeffs are fitted separately and will write out
                param = param +'_{}:\t'.format(int(priorinfo._filter_idx[i]))
            else:
                param += ':\t'
            print('{}\t {} ± {}'.format(param, value, unc))

    # Deal with 'single' fitting LD param mode
    if priorinfo.ld_fit_method == 'single':
        # We've only fitted one wavelengths' LD coeffs:
        # Estimate the rest of the limb darkening values and write.

        # Get the fitted values in a usable format
        best_single_ld = np.zeros(len(priorinfo.limb_dark_coeffs))
        single_uncertainty = np.zeros(len(priorinfo.limb_dark_coeffs))

        for i, ui in enumerate(priorinfo.limb_dark_coeffs):
            x = np.where(np.array(priorinfo.fitting_params) == ui)[0]
            best_single_ld[i] = best[x]
            single_uncertainty = results.uncertainties[x]

        # Now do the estimation
        best_ld_params = priorinfo.ld_handler.ldtk_estimate(best_single_ld)

        # We estimate the errors by using the ratios
        estim_errors = priorinfo.ld_handler.ldtk_estimate(single_uncertainty)

        # Loop through each filter and write out the values
        for i, fi in enumerate(best_ld_params):
            for j, qj in enumerate(priorinfo.limb_dark_coeffs):
                param = qj + '_{}:\t'.format(i)
                value = round(fi[j], 6)
                unc = round(estim_errors[i, j], 6)

                print('{}\t {} ± {}'.format(param, value, unc))

    best_chi2 = - results.logl.max()

    print('chi2:\t\t {}'.format(round(best_chi2, 5)))
    print('red chi2:\t {}'.format(round(best_chi2/n_dof, 5)))


def save_final_light_curves(lightcurves, priorinfo, results,
                            folder='./final_light_curves'):
    '''
    Applies detrending and normalisation to each light curve and saves to .csv

    Parameters
    ----------
    lightcurves : array_like, shape (n_telescopes, n_filters, n_epochs)
        An array of LightCurves. If no data exists for a point in the array
        then the entry should be `None`.
    priorinfo : transitfit.priorinfo.PriorInfo
        The PriorInfo object
    results : dynesty.results.Results
        The Dynesty results object, but must also have weights, cov and
        uncertainties as entries.
    folder : str, optional
        The folder to save the files to. Default is './final_light_curves'
    '''
    # Get the best values
    best = results.samples[np.argmax(results.logl)]
    best_dict = priorinfo._interpret_param_array(best)

    # Get the array of detrending coeffs:
    if priorinfo.detrend:
        # We need to combine the detrending coeff arrays into one
        # Each entry should be a list containing all the detrending
        # coefficients to trial.
        d = np.full(lightcurves.shape, None, object)

        for i in np.ndindex(d.shape):
            for coeff in priorinfo.detrending_coeffs:
                if best_dict[coeff][i] is not None:
                    if d[i] is None:
                        d[i] = [best_dict[coeff][i]]
                    else:
                        d[i].append(best_dict[coeff][i])

    os.makedirs(folder, exist_ok=True)

    # Loop over each light curve and apply detrending and normalisation
    for i in np.ndindex(lightcurves.shape):
        if lightcurves[i] is not None:

            # Calculate the detrended light curve
            norm = best_dict['norm'][i]

            if priorinfo.detrend:
                detrended_flux, detrended_errors = lightcurves[i].detrend_flux(d[i], norm)
            else:
                detrended_flux, detrended_errors = lightcurves[i].detrend_flux(None, norm)

            # Calculate the value of the best fit light curve at the same times
            # First we set up the parameters
            params = batman.TransitParams()
            params.t0 = best_dict['t0']
            params.per = best_dict['P']
            params.rp = best_dict['rp'][i[1]]
            params.a = best_dict['a']
            params.inc = best_dict['inc']
            params.ecc = best_dict['ecc']
            params.w = best_dict['w']
            params.limb_dark = priorinfo.limb_dark
            params.u = np.array([best_dict[key] for key in priorinfo.limb_dark_coeffs]).T[i[1]]

            m_sample_times = batman.TransitModel(params, lightcurves[i].times)
            time_wise_best_curve = m_sample_times.light_curve(params)

            write_dict = []
            for j, tj in enumerate(lightcurves[i].times):
                #print(i)
                write_dict.append({'Time' : tj,
                                   'Normalised flux' : detrended_flux[j],
                                   'Uncertainty' : detrended_errors[j],
                                   'Best fit curve' : time_wise_best_curve[j]})

            fname = 't{}_f{}_e{}_detrended.csv'.format(i[0], i[1], i[2])
            with open(os.path.join(folder, fname), 'w') as f:
                columns = ['Time', 'Normalised flux', 'Uncertainty', 'Best fit curve']
                writer = csv.DictWriter(f, columns)
                writer.writeheader()
                writer.writerows(write_dict)
