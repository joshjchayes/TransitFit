'''
file_reader

Module to help make using csv files super easy as inputs
'''

import numpy as np
import pandas as pd
from .priorinfo import PriorInfo, _prior_info_defaults

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
    with columns in the order time, depth, errors

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


def read_priors_file(path):
    '''
    If given a csv file containing priors, will produce a PriorInfo object
    based off the given values

    Columns should me in the order
    -----------------------------------------------------------------
    |  key  |   best  |  low_lim  |   high_lim  |  light_curve_num  |
    -----------------------------------------------------------------

    If the parameter is invariant across the light curves, leave
    light_curve_num blank.
    '''

    table = pd.read_csv(path).values

    # Work out how mnay light curves are being used.
    num_light_curves = int(np.nanmax(table[:, -1]) + 1)
    print(num_light_curves)

    #######################################################
    # Set up the default dict to initialise the PriorInfo #
    #######################################################
    default_prior_dict = {}
    default_prior_dict['rp'] = np.full(num_light_curves, np.nan)
    default_prior_dict['t0'] = np.full(num_light_curves, np.nan)
    default_prior_dict['num_light_curves'] = num_light_curves

    for row in table:
        key, best, low, high, lcnum = row
        if key in ['rp', 't0']:
            default_prior_dict[key][int(lcnum)] = best
        else:
            default_prior_dict[key] = best

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

    prior_info = PriorInfo(default_prior_dict)

    #################################
    # Now add in the actual priors! #
    #################################
    for row in table:
        key, best, low, high, lcnum = row
        if key in ['rp', 't0']:
            prior_info.add_uniform_fit_param(key, best, low, high, int(lcnum))
        else:
            prior_info.add_uniform_fit_param(key, best, low, high)

    return prior_info
