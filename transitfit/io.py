'''
Module to deal with all the reading in of inputs, as well as printing final results to terminal
'''

import numpy as np
import pandas as pd
from .priorinfo import PriorInfo, _prior_info_defaults, setup_priors
import dynesty
import csv
import os
from ._utils import validate_variable_key, AU_to_host_radii
from .lightcurve import LightCurve
import batman
import os

#############################################################
#                         PRIORS                            #
#############################################################
def read_priors_file(path, n_telescopes, n_filters, n_epochs,
                     limb_dark='quadratic', filter_indices=None, folded=False,
                     folded_P=None, folded_t0=None, host_radius=None,
                     allow_ttv=None, lightcurves=None, suppress_warnings=False):
    '''
    If given a csv file containing priors, will produce a PriorInfo object
    based off the given values

    Parameters
    ----------
    path : str
        Path to .csv file containing the priors.

        Columns should me in the order
        --------------------------------------------------------
        |  key  |   mode  |  input_A  |   input_B   |  filter  |
        --------------------------------------------------------

        The available modes and the expected values of inputs A and B are:

        - 'uniform': input A should be lower limit, input B should be upper
                     limit
        - 'gaussian': input_A should be mean, input_B should be standard
                      deviation.
        - 'fixed': input_A should be the fixed value. input_B is not used and
                   should be left blank.

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
    filter_indices : array_like, optional
        If provided, will only initialise fitting for parameters which are
        relevant to the filter indices given. Note that this will result in
        a difference between the filter indices used at the top, user level and
        those used within this PriorInfo
    folded : bool, optional
        If True, will not initialise P or t0 from the priors list. Instead will
        use folded_P and folded_t0 to set fixed values. Default is False
    folded_P : float, optional
        Required if folded is True. This is the period that the light curves
        are folded to
    folded_t0 : float, optional
        Required if folded is True. This is the t0 that the light curves are
        folded to
    host_radius : float, optional
        The host radius in Solar radii.
        If this is provided, then will assume that the orbital separation is
        given in AU rather than host radii and will convert the values
        accordingly.

    Notes
    -----
    Detrending currently cannot be initialised in the prior file. It will be
    available as a kwarg in the pipeline function
    '''
    priors_list = pd.read_csv(path).values

    return parse_priors_list(priors_list, n_telescopes, n_filters, n_epochs, limb_dark, filter_indices, folded, folded_P, folded_t0, host_radius, allow_ttv, lightcurves, suppress_warnings)

def parse_priors_list(priors_list, n_telescopes, n_filters,
                      n_epochs, ld_model, filter_indices=None, folded=False,
                      folded_P=None, folded_t0=None, host_radius=None,
                      allow_ttv=False, lightcurves=None, suppress_warnings=False):
    '''
    Parses a list of priors to produce a PriorInfo with all fitting parameters
    initialised.

    Parameters
    ----------
    priors_list : array_like, shape(X, 5)
        A list of prior information for each variable to be fitted. Can also
        set fixed values by setting the low and high values to `None` or
        `np.nan`. Each row should be of the form
        [key, mode, input A, input B, filter index]
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
    filter_indices : array_like, optional
        If provided, will only initialise fitting for parameters which are
        relevant to the filter indices given. Note that this will result in
        a difference between the filter indices used at the top, user level and
        those used within this PriorInfo
    folded : bool, optional
        If True, will not initialise P or t0 from the priors list. Instead will
        use folded_P and folded_t0 to set fixed values. Default is False
    folded_P : float, optional
        Required if folded is True. This is the period that the light curves
        are folded to
    folded_t0 : float, optional
        Required if folded is True. This is the t0 that the light curves are
        folded to
    host_radius : float, optional
        The host radius in Solar radii. If this is provided, then will assume
        that the orbital separation is given in AU rather than host radii and
        will convert the values accordingly

    Returns
    -------
    priors : PriorInfo
        The fully initialised PriorInfo which can then be used in fitting.
    '''
    if folded:
        if folded_P is None:
            raise ValueError('folded_P must be provided for folded prior mode')
        if folded_t0 is None:
            raise ValueError('folded_t0 must be provided for folded prior mode')

    if allow_ttv and lightcurves is None:
        raise ValueError('lightcurves must be provided if allow_ttv is True')

    # Make a dictionary of the defaults
    priors_dict = {}

    if filter_indices is None:
        # We will use all filters
        filter_indices = np.arange(n_filters)

    rp_count = 0
    used_filters = []
    for row in priors_list:
        # First check the key and correct if possible
        row[0] = validate_variable_key(row[0])

        # Now add to the priors_dict
        if row[0] in ['rp']:
            # We have to deal with extracting particular filters

            if row[-1] in filter_indices:
                priors_dict[row[0]] = np.append(row[2:-1], rp_count)
                rp_count += 1
                used_filters.append(row[-1])

        else:
            if row[0] == 'a' and host_radius is not None:
                # Convert the inputs to host radius units
                row[2] = AU_to_host_radii(row[2], host_radius)
                row[3] = AU_to_host_radii(row[3], host_radius)

            priors_dict[row[0]] = row[2:]

    # We  have to convert between the global filter indexing and an
    # internal filter indexing here.
    filter_conversion = {ai : i for i, ai in enumerate(used_filters)}
    # Each key is the global value, and converts to the internal value.

    ##############################
    # Make the default PriorInfo #
    ##############################

    # Need to check if any variables are missing from the default prior
    for key in _prior_info_defaults:
        if key not in priors_dict:
            # Has not been specified in the priors list
            priors_dict[key] = [_prior_info_defaults[key]]

    # Update priors_dict values if using folded curves
    if folded:
        #
        priors_dict['P'] = [folded_P]
        priors_dict['t0'] = [folded_t0]

    # Now make the basic PriorInfo
    priors = setup_priors(priors_dict['P'][0],
                          priors_dict['t0'][0],
                          priors_dict['a'][0],
                          priors_dict['rp'][0],
                          priors_dict['inc'][0],
                          priors_dict['ecc'][0],
                          priors_dict['w'][0],
                          ld_model, n_telescopes, n_filters, n_epochs,
                          priors_dict['q0'][0],
                          priors_dict['q1'][0],
                          priors_dict['q2'][0],
                          priors_dict['q3'][0],
                          allow_ttv, lightcurves)

    ##########################
    # Initialise the fitting #
    ##########################
    for ri, row in enumerate(priors_list):
        key, mode, inputA, inputB, filt = row
        mode = mode.strip()

        if pd.notna(filt) and filt not in filter_indices:
            # Skip this parameter since it's not in the filters we are
            # interested in
            pass

        else:
            # Convert the filter index
            try:
                filt = filter_conversion[filt]
            except KeyError:
                if pd.isna(filt):
                    # This was excepted because filt was not specified
                    filt = None
                else: raise

            if key == 'a' and host_radius is not None:
                # Convert the inputs to host radius units
                row[2] = AU_to_host_radii(row[2], host_radius)
                row[3] = AU_to_host_radii(row[3], host_radius)

            if key in ['P', 't0'] and folded:
                # Skip P and t0 for folded mode - we aren't fitting them
                pass

            elif key in ['P'] and allow_ttv:
                # We can't fit period if we are using ttv mode - skip it
                if not suppress_warnings:
                    print("WARNING: Ignoring P fitting due to ttv mode. It is recommended to specify P as 'Fixed' in the input file, else TransitFit will default to the value given with Input A.")
                pass

            elif mode.lower() in ['fixed', 'f', 'constant', 'c']:
                # Not being fitted. Default value was specified.
                pass

            elif mode.lower() in ['uniform', 'unif', 'u']:
                # Uniform fitting
                if key in ['t0'] and allow_ttv:
                    for li in np.ndindex(lightcurves.shape):
                        # Loop through each lightcurve and initialise t0
                        if lightcurves[li] is not None:
                            priors.add_uniform_fit_param(key, inputA, inputB, li[0], li[1], li[2])
                else:
                    priors.add_uniform_fit_param(key, inputA, inputB, filter_idx=filt)

            elif mode.lower() in ['gaussian', 'gauss', 'normal', 'norm', 'g']:
                # Gaussian fitting
                if key in ['t0'] and allow_ttv:
                    for li in np.ndindex(lightcurves.shape):
                        # Loop through each lightcurve and initialise t0
                        if lightcurves[li] is not None:
                            priors.add_gaussian_fit_param(key, inputA, inputB, li[0], li[1], li[2])
                else:
                    priors.add_gaussian_fit_param(key, inputA, inputB, filter_idx=filt)

            else:
                raise ValueError('Unrecognised fiting mode {} in input row {}. Must be any of "uniform", "gaussian", or "fixed"'.format(mode, ri))

    return priors

#############################################################
#                       DATA FILES                          #
#############################################################
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

def _read_data_csv(path, usecols=None):
    '''
    Given a path to a csv with columns [time, flux, errors], will get
    all the data in a way which can be used by the Retriever

    '''
    # Read in with pandas
    data = pd.read_csv(path, usecols=usecols)

    # Extract the arrays
    times, flux, errors = data.values.T

    non_nan = np.invert(np.any(pd.isna(data.values.T), axis=0))

    return times[non_nan], flux[non_nan], errors[non_nan]

def _read_data_txt(path, skiprows=0, usecols=None, delimiter=' '):
    '''
    Reads a txt data file with columns
    '''
    try:
        data = pd.read_csv(path, usecols=usecols, dtype=float, delimiter=delimiter)
        times, flux, errors = data.values.T
        non_nan = np.invert(np.any(pd.isna(data.values.T), axis=0))
        return times[non_nan], flux[non_nan], errors[non_nan]

    except Exception as e:
         times, flux, errors = np.loadtxt(path, skiprows=skiprows, usecols=usecols).T
         return times, flux, errors

         raise e

def read_data_file(path, skiprows=0, folder=None, usecols=None, delimiter=None):
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
    folder : str or None, optional
        If not None, this folder will be prepended to all the paths. Default is
        None.
    usecols : int or sequence, optional
        Which columns to read, with 0 being the first. For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
        These should be given in the order time, flux, uncertainty
        The default, None, results in all columns being read.

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
        times, flux, errors = _read_data_csv(os.path.join(folder, path), usecols)
    elif path[-4:] == '.txt':
        times, flux, errors = _read_data_txt(os.path.join(folder, path), skiprows, usecols)
    else:
        raise ValueError(f'Data files must be .csv or .txt, not {path[-4:]}')

    return times, flux, errors

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
            times, flux, errors = read_data_file(data_path_array[i], skiprows)

            lightcurves[i] = LightCurve(times, flux, errors, i[0], i[1], i[2])

    return lightcurves

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
    column_array = np.full((n_telescopes, n_filters, n_epochs), None, object)

    # Populate the blank array
    for row in data_path_list:
        p, i, j, k, l = row
        data_path_array[i, j, k] = p
        detrending_index_array[i, j, k] = l

    return data_path_array, detrending_index_array

#############################################################
#                         FILTERS                           #
#############################################################
def parse_filter_list(filter_list, delimiter=None, unit='nanometers'):
    '''
    Parses a list of filter information into a usable form for the `'filters'`
    argument in PriorInfo.fit_limb_darkening.

    Parameters
    ----------
    filter_list : array_like, shape (n_filters, 3)
        The information on the filters. Each row should contain
        [filter_index, low wavelength/file path, high wavelength]
        The filter indices should refer to the indices used in the priors file.
        Wavelengths should be in nm.
    delimiter : str, optional
        The delimiter used in filter profile files. Default is None, which
        automatically detects using csv.Sniffer

    Returns
    -------
    filter_info : np.array, shape (n_filters, 2)
        The filter information pass to PriorInfo.fit_limb_darkening.
    '''
    filter_list = np.array(filter_list, dtype=object)

    # How many filters are there?
    n_filters = int(filter_list[:,0].max() + 1)

    # Make a blank array to populate with the filter limits
    filter_info = np.zeros((n_filters, 2), object)

    # Now populate!
    for i in range(n_filters):
        fidx, input_A, input_B = filter_list[i]

        # Check if path, name, or limits are provided:
        try:
            input_A = float(input_A)
            input_B = float(input_B)
            limits_provided=True
        except:
            limits_provided=False

        if limits_provided:
            # box limits provided
            filter_info[i] = input_A, input_B
        else:
            # Get either the path from input or replace path for provided filters
            path, unit = get_filter_path(filter_list[i, 1], unit)
            # Load in the filter profile
            filter_profile = pd.read_csv(path, sep=delimiter, dtype=float).values.T

            # Since filter wavelengths need to be in nm, we should convert them
            # from angstroms if necessary:
            if unit.lower() in ['nm', 'nanometers']:
                factor = 1
            elif unit.lower() in ['angstroms', 'angstrom', 'a']:
                factor = 0.1
            elif unit.lower() in ['m','meters','metres']:
                factor = 1e-9
            else:
                raise ValueError(f'Unrecognised unit {unit} given for filter profiles. TransitFit can only recognise "m", "nm", or "angstroms".')
            filter_info[i,0] = filter_profile[0] * factor
            filter_info[i,1] = filter_profile[1]

    return filter_info

def read_filter_info(path, delimiter=None, unit='nanometers'):
    '''
    Reads in information on the filters from .csv file and puts them in a
    format which can be passed to the ``filters`` argument in
    PriorInfo.fit_limb_darkening.

    Parameters
    ----------
    path : str
        Path to the .csv file containing the information on the filters. This
        file should have three columns:

        -------------------------------------------------
        |   filter_idx  |   low_wl_or_path  |   high_wl |
        -------------------------------------------------

        Filters can either be specified as uniform between low_wl and high_wl,
        or a full profile can be passed through a file. Filter profile files
        must be two columns, giving wavelength and transmission fraction in
        range [0,1].
        The filter indices should refer to the indices used in the priors file.
        All wavelengths should be in nm.

    delimiter : str, optional
        The delimiter used in filter profile files. Default is None, which
        automatically detects using csv.Sniffer

    Returns
    -------
    filter_info : np.array, shape (n_filters, 2)
        The filter information pass to PriorInfo.fit_limb_darkening.

    '''

    info = pd.read_csv(path).values

    return parse_filter_list(info, delimiter)

def get_filter_path(input_str, unit):
    '''
    Checks if a path or name of a default provided filter is given.

    Parameters
    ----------
    input : str
        The input string from the filter file input

    Returns
    -------
    path : str
        The path to the relevant filter
    '''
    input_str = input_str.strip()

    if os.path.exists(input_str):
        return input_str, unit

    filter_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # Johnson UBVRI
    if input_str == 'U':
        return os.path.join(filter_folder, 'filters/JohnsonU.csv'), 'angstroms'
    if input_str == 'B':
        return os.path.join(filter_folder, 'filters/JohnsonB.csv'), 'angstroms'
    if input_str == 'V':
        return os.path.join(filter_folder, 'filters/JohnsonV.csv'), 'angstroms'
    if input_str == 'R':
        return os.path.join(filter_folder, 'filters/CousinsR.csv'), 'angstroms'
    if input_str == 'I':
        return os.path.join(filter_folder, 'filters/CousinsI.csv'), 'angstroms'

    # SDSS filters:
    if input_str == 'g':
        return os.path.join(filter_folder, 'filters/SLOAN_g.csv'), 'angstroms'
    if input_str == 'g_prime':
        return os.path.join(filter_folder, 'filters/SLOAN_gprime.csv'), 'angstroms'
    if input_str == 'i':
        return os.path.join(filter_folder, 'filters/SLOAN_i.csv'), 'angstroms'
    if input_str == 'i_prime':
        return os.path.join(filter_folder, 'filters/SLOAN_iprime.csv'), 'angstroms'
    if input_str == 'r':
        return os.path.join(filter_folder, 'filters/SLOAN_r.csv'), 'angstroms'
    if input_str == 'r_prime':
        return os.path.join(filter_folder, 'filters/SLOAN_rprime.csv'), 'angstroms'
    if input_str == 'u':
        return os.path.join(filter_folder, 'filters/SLOAN_u.csv'), 'angstroms'
    if input_str == 'u_prime':
        return os.path.join(filter_folder, 'filters/SLOAN_uprime.csv'), 'angstroms'
    if input_str == 'z':
        return os.path.join(filter_folder, 'filters/SLOAN_z.csv'), 'angstroms'
    if input_str == 'z_prime':
        return os.path.join(filter_folder, 'filters/SLOAN_zprime.csv'), 'angstroms'

    # Kepler and TESS
    if input_str.lower() == 'kepler':
        return os.path.join(filter_folder, 'filters/Kepler.csv'), 'angstroms'
    if input_str.lower() == 'tess':
        return os.path.join(filter_folder, 'filters/TESS.csv'), 'angstroms'


    raise ValueError(f'Unable to convert "{input_str}" into a path to a filter file')


#############################################################
#                         OUTPUTS                           #
#############################################################
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
    for i, param_info in enumerate(priorinfo.fitting_params):
        param, tidx, fidx, eidx = param_info
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
                param = param +'_{}:\t'.format(int(fidx))
            #elif param in ['t0']:
            #    param = param +'_{}'.format(int(priorinfo._epoch_idx[i]))
            elif (param in priorinfo.detrending_coeffs + ['norm']) or (param in['t0'] and priorinfo.allow_ttv):
                param = param + '_t{}_f{}_e{}:'.format(int(tidx),int(fidx), int(eidx))
            elif param in priorinfo.limb_dark_coeffs and priorinfo.ld_fit_method in ['independent', 'coupled']:
                # All the LD coeffs are fitted separately and will write out
                param = param +'_{}:\t'.format(int(fidx))
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
                            folder='./final_light_curves', folded=False):
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
    folded : bool, optional
        If True, will assume that the lightcurves provided are folded and
        will change the filenames accordingly. Default is False
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
            for coeff in np.ravel(priorinfo.detrending_coeffs):
                if best_dict[coeff][i] is not None:
                    if d[i] is None:
                        d[i] = [best_dict[coeff][i]]
                    else:
                        d[i].append(best_dict[coeff][i])

    os.makedirs(folder, exist_ok=True)

    # Loop over each light curve and apply detrending and normalisation
    for i in np.ndindex(lightcurves.shape):
        if lightcurves[i] is not None:

            telescope_idx = lightcurves[i].telescope_idx
            filter_idx = lightcurves[i].filter_idx
            epoch_idx = lightcurves[i].epoch_idx

            # Calculate the detrended light curve
            norm = best_dict['norm'][i]

            if priorinfo.detrend:
                detrended_flux, detrended_errors = lightcurves[i].detrend_flux(d[i], norm)
            else:
                detrended_flux, detrended_errors = lightcurves[i].detrend_flux(None, norm)

            # Calculate the value of the best fit light curve at the same times
            # First we set up the parameters
            params = batman.TransitParams()
            params.t0 = best_dict['t0'][i]
            params.per = best_dict['P'][i]
            params.rp = best_dict['rp'][i]
            params.a = best_dict['a'][i]
            params.inc = best_dict['inc'][i]
            params.ecc = best_dict['ecc'][i]
            params.w = best_dict['w'][i]
            params.limb_dark = priorinfo.limb_dark

            if priorinfo.fit_ld:
                best_q = np.array([best_dict[key][i] for key in priorinfo.limb_dark_coeffs])
            else:
                q = np.array([priorinfo.priors[key][i] for key in priorinfo.limb_dark_coeffs])
                for j in np.ndindex(q.shape):
                    q[j] = q[j]
                best_q = q

            # Convert from q to u
            params.u = priorinfo.ld_handler.convert_qtou(*best_q)

            m_sample_times = batman.TransitModel(params, lightcurves[i].times)
            time_wise_best_curve = m_sample_times.light_curve(params)

            # Calculate phase for each point in the curve
            phase = lightcurves[i].get_phases(best_dict['t0'][i], best_dict['P'][i])

            write_dict = []
            for j, tj in enumerate(lightcurves[i].times):
                write_dict.append({'Time' : tj,
                                   'Phase' : phase[j],
                                   'Normalised flux' : detrended_flux[j],
                                   'Uncertainty' : detrended_errors[j],
                                   'Best fit curve' : time_wise_best_curve[j]})

            if folded:
                fname = 'filter_{}_FOLDED.csv'.format(filter_idx)
            else:
                fname = 't{}_f{}_e{}_detrended.csv'.format(telescope_idx, filter_idx, epoch_idx)

            with open(os.path.join(folder, fname), 'w') as f:
                columns = ['Time', 'Phase', 'Normalised flux', 'Uncertainty', 'Best fit curve']
                writer = csv.DictWriter(f, columns)
                writer.writeheader()
                writer.writerows(write_dict)
