'''
detrending_funcs.py

Some detrending functions for use with LightCurve
'''

import numpy as np


class NthOrderDetrendingFunction:
    def __init__(self, order):
        '''
        Arbitrary order detrending function which conserves flux
        '''
        order = int(order)
        if not order > 0:
            raise ValueError('Order must be greater than 0')

        self.order = order


    def __call__(self, times, *args):
        '''
        Returns detrending values for the times
        '''
        if not len(args) == self.order:
            raise ValueError('Number of arguments {} supplied does not match the order {} of the detrending function.'.format(len(args), self.order))

        vals = np.zeros(len(times))
        for i in range(0, self.order):
            vals += args[i] * (times ** (i+1) - sum(times ** (i+1))/len(times))

        return vals

linear = NthOrderDetrendingFunction(1)
quadratic = NthOrderDetrendingFunction(2)
