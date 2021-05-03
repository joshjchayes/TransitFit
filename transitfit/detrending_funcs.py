'''
Some detrending functions for use with LightCurve
'''

import numpy as np


class NthOrderDetrendingFunction:
    '''
    Arbitrary order detrending function which conserves flux

    Parameters
    ----------
    order : int
        The detrending function order. Must be greater than 0.
    '''
    def __init__(self, order):

        order = int(order)
        if not order > 0:
            raise ValueError('Order must be greater than 0')

        self.order = order


    def __call__(self, lightcurve, *args):
        '''
        Detrends the lightcurve and returns the flux values
        '''
        if not len(args) == self.order:
            raise ValueError('Number of arguments {} supplied does not match the order {} of the detrending function.'.format(len(args), self.order))

        vals = np.zeros(len(lightcurve.times))
        for i in range(0, self.order):
            vals += args[i] * (lightcurve.times ** (i+1) - np.mean(lightcurve.times ** (i+1)))

        return vals

linear = NthOrderDetrendingFunction(1)
quadratic = NthOrderDetrendingFunction(2)
