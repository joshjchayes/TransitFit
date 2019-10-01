'''
_utils.py

This is a series of utility functions for TransitFit, including input validation
'''

import numpy as np

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


def calculate_logg(host_mass, host_radius):
    '''
    Calculates log10(g) for a host in units usable by TransitFit. g is in


    Parameters
    ----------
    host_mass : float
        The host mass in solar masses
    host_radius : float
        The host radius in solar radii
    '''

    m_sun = 1.989e30
    r_sun = 6.957e8
    G = 6.674e-11

    return np.log10((host_mass * m_sun * G)/((host_radius * r_sun) ** 2) * 100)
