'''
detrending_funcs.py

Some detrending functions for use with LightCurve
'''

import numpy as np

def linear(times, u1):
    '''
    A linear detrend. Generates detrending values at the given times using

        detrend_value = u1 * times

    Parameters
    ----------
    times : array_like
        The times to generate detrending values for
    u1 : float
        First coefficient
    u2 : float
        Second coefficient
    '''

    return u1 * times

def quadratic(times, u1, u2):
    '''
    Quadratic detrend

    detrend_values = u1 * times^2 + u2 * times
    '''
    return u1 * times**2 + u2 * times

def sinusoidal(times, u1, u2, u3):
    '''
    Sinusoidal detrend

    detrend_values = u1 * sin(u2 * times + u3)

    '''
    return u1 * np.sin(u2 * times + u3)
