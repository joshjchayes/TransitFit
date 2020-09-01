'''
_utils.py

This is a series of utility functions for TransitFit, including input validation
'''

import numpy as np
from .lightcurve import LightCurve
from collections.abc import Iterable

def validate_lightcurve_array_format(arr):
    '''
    Used to check that the input LightCurve data arrays are compatible with
    TransitFit, and corrects for any simple mistakes in the formatting.
    '''

    arr = np.array(arr)

    if arr.ndim == 0:
        # We have a single LightCurve
        arr = arr.reshape(1,1,1)
    elif arr.ndim == 1:
        # This is either a single light curve or we are missing axes!
        # We can check this by looking for any entries with None: if there
        # are any, then we are missing an axis and should raise an error as
        # we cannot tell which of telescope, wavelength or epoch is missing.
        if np.any(arr[arr] == None):
            raise ValueError('You appear to be missing either the telescope, wavelength or epoch axis in your input LightCurve array. Please check the format and try again')
        arr = arr.reshape(1,1,1)
    elif arr.ndim == 2:
        # We are missing an axis.
        raise ValueError('You appear to be missing either the telescope, wavelength or epoch axis in your input LightCurve array. Please check the format and try again')

    elif arr.ndim > 3:
        raise ValueError('There are too many axes in your input LightCurve array! There should only be three, in the order (telescope, filter, epoch)')

    return arr


def validate_data_format(a1, a2, a3):
    '''
    This function is to be used to check that the 3 input data arrays
    are compatible, and also checks for and corrects any simple
    mistakes in the formatting.

    One of the big uses for this is to ensure that the arrays are numpy
    arrays and also to allow single light curves to be passed with
    minimal effort

    Returns
    -------
    a1 : np.array
        The array a1 but corrected for use in TransitFit
    a2 : np.array
        The array a2 but corrected for use in TransitFit
    a3 : np.array
        The array a3 but corrected for use in TransitFit
    '''

    # First we check the shapes of each array and set them as a 2D
    # numpy array where each entry is data series for the light curve
    np_arrays = {'a1':np.array(a1, object),
                 'a2':np.array(a2, object),
                 'a3':np.array(a3, object)}

    for arr in np_arrays:
        if np_arrays[arr].ndim > 2:
            raise ValueError('Array {} has {} > 2 dimensions!'.format(arr, np_arrays[arr].ndim))
        if np_arrays[arr].ndim == 0:
            raise ValueError("Array {} has 0 dimensions! It looks like you've only given me one data point!")
        if np_arrays[arr].ndim == 1:
            # This is either a single light curve or we are missing an axis!
            # We can check this by looking for any entries with None: if there
            # are any, then we are missing an axis and should raise an error as
            # we cannot tell which of wavelength or epoch is missing
            if np.any(np_arrays[arr] == None):
                raise ValueError('Array {}: You appear to be missing either the wavelength or epoch axis in your input array. Please check the format and try again'.format(arr))
            np_arrays[arr] = np_arrays[arr].reshape(2, np_arrays[arr].shape[0])

    # Now check that they are all the same shape!!
    if not np_arrays['a1'].shape == np_arrays['a2'].shape == np_arrays['a3'].shape:
        raise ValueError('I was unable to cast all the input arrays into the same shape! I only managed to get the shapes {}, {}, {}'.format(np_arrays['a1'].shape, np_arrays['a2'].shape, np_arrays['a3'].shape))

    # Check that all the 'None's are in the same place:
    if np.any(np_arrays['a1'] == None):
        if not np.where(np_arrays['a1'] == None) == np.where(np_arrays['a2']==None) == np.where(np_arrays['a3']==None):
            raise ValueError("The None values in your input arrays don't match! Check this then try again!")

    # Check that each data set is the same length, since each light
    # curve will have a different number of data points
    for i, row in enumerate(np_arrays['a1']):
        for j, column in enumerate(np_arrays['a1']):
            if not len(np_arrays['a1']) == len(np_arrays['a2']) == len(np_arrays['a3']):
                raise ValueError('There is an issue with the number of data points you gave for index ({}, {}) in your input arrays. Check that all each of these are the same length and then try again.'.format(i, j))

    return np_arrays['a1'], np_arrays['a2'], np_arrays['a3']


def validate_variable_key(key):
    '''
    Checks that a key is valid for use with PriorInfo, and corrects when it
    is obvious what is meant. Raises KeyError if unable to correct.

    '''
    if key.lower() in ['p','period']:
        return 'P'
    if key.lower() in ['rp','r_p','radius','planet_radius', 'planet radius']:
        return 'rp'
    if key.lower() in ['a', 'semimajor axis', 'semimajor_axis']:
        return 'a'
    if key.lower() in ['inc','inclination']:
        return 'inc'
    if key.lower() in ['t0', 't_0']:
        return 't0'
    if key.lower() in ['ecc','e','eccentricity']:
        return 'ecc'
    if key.lower() in ['w', 'periastron', 'longitude_of_periastron', 'longitude of periastron']:
        return 'w'
    if key.lower() in ['q0']:
        return 'q0'
    if key.lower() in ['q1']:
        return 'q1'
    if key.lower() in ['q2']:
        return 'q2'
    if key.lower() in ['q3']:
        return 'q3'

    raise KeyError('Unable to recognise variable name {}'.format(key))


def calculate_logg(host_mass, host_radius):
    '''
    Calculates log10(g) for a host in units usable by TransitFit.
    g is in cm s^-2

    Parameters
    ----------
    host_mass : tuple
        The host mass in solar masses and the uncertainty
    host_radius : tuple
        The host radius in solar radii and the uncertainty
    '''


    m_sun = 1.989e30
    r_sun = 6.957e8
    G = 6.674e-11

    m = host_mass[0] * m_sun
    err_m = host_mass[1] * m_sun
    r = host_radius[0] * r_sun
    err_r = host_radius[1] * r_sun

    g = m * G * r ** -2 * 100 # factor 100 converts to cm s^-2
    err_g = G/(r ** 2) * np.sqrt(err_m ** 2 + (4 * m**2 * err_r ** 2)/r**2) * 100

    logg = np.log10(g)
    err_logg = err_g / (g * np.log(10))

    return logg, err_logg


def get_normalised_weights(results):
    '''
    Obtains normalised weights for the Dynesty results

    Parameters
    ----------
    results : dynesty.results.Results
        The Dynesty results object

    Returns
    -------
    weights : np.array, shape (n_iterations,)
        The normalised weights for each sample set
    '''

    return np.exp(results.logwt - results.logwt.max())/np.sum(np.exp(results.logwt - results.logwt.max()))


def get_covariance_matrix(results):
    '''
    Gets a covariance matrix from Dynesty results

    Parameters
    ----------
    results : dynesty.results.Results
        The Dynesty results object, but must also have weights as an entry

    Returns
    -------
    cov : np.array, shape (ndims, ndims)
        The covariance matrix for the results object.
    '''

    # Calculate a covariance matrix using numpy
    cov = np.cov(results.samples, rowvar=False, aweights=results.weights)

    return cov

def weighted_avg_and_std(values, weights, axis=-1):
    '''
    Calculates the weighted average and error on some data.

    axis defaults to the last one (-1)
    '''
    if not isinstance(values, Iterable):
        values = [values]
    if not isinstance(weights, Iterable):
        weights = [weights]

    values = np.array(values)
    weights = np.array(weights)

    shape = values.shape
    if len(shape) == 1:
        # We have only been given one value - return it
        return values[0], weights[0]

    average = np.average(values, weights=weights, axis=axis)
    # Reshape average
    shape = shape[:-1] + (1,)
    average = average.reshape(shape)

    variance = np.average((values-average)**2, weights=weights, axis=axis).reshape(shape)


    return average, np.sqrt(variance)

def AU_to_host_radii(a, R):
    '''
    Converts a number in AU to a value in host radii when given the host
    radius R in Solar radii
    '''
    AU = 1.495978707e11
    R_sun = 6.957e8

    return (a * AU) / (R * R_sun)


def split_lightcurve_file(path, t0, P, new_base_fname=None):
    '''
    Splits a multi-epoch lightcurve data file into multiple single-epoch files
    and saves these.
    '''
    from .io import read_data_file

    # Load the full data file as a LightCurve
    full_lightcurve = LightCurve(*read_data_file(path))

    # Split the full curve into individual epochs
    single_epoch_curves = full_lightcurve.split(t0, P)

    # Now save all the new curves
    if new_base_fname is None:
        new_base_fname = 'split_curve'

    for i, curve in enumerate(single_epoch_curves):
        fname = new_base_fname + '_{}'.format(i)
        curve.save(fname)

    return single_epoch_curves
