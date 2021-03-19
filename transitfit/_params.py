'''
A class for parameters in TransitFit which can be retrieved. These are used
by the PriorInfo to determine dimensionality etc.

'''
from scipy.special import erfinv
import numpy as np

class _Param:
    def __init__(self, value, uncertainty=None):

        self.default_value = value
        self.low_lim=None
        self.high_lim=None
        self.uncertainty=uncertainty

    def from_unit_interval(self, u):
        raise NotImplementedError


class _UniformParam(_Param):
    def __init__(self, low_lim, high_lim, negative_allowed=True):
        if low_lim >= high_lim:
            raise ValueError('low_lim >= high_lim')

        super().__init__((high_lim + low_lim)/2)
        self.low_lim = low_lim
        self.high_lim = high_lim
        self.negative_allowed = negative_allowed

    def from_unit_interval(self, u):
        '''
        Function to convert value u in range (0,1], will convert to a value to
        be used by Batman
        '''
        if u > 1 or u < 0:
            raise ValueError('u must satisfy 0 < u < 1. ')
        val = u * (self.high_lim - self.low_lim) + self.low_lim
        if self.negative_allowed:
            return val
        return abs(val)

class _GaussianParam(_Param):
    def __init__(self, best, sigma, negative_allowed=True):
        '''
        A GaussianParam is one which is fitted using a Gaussian prior (normal)
        distribution.
        '''

        super().__init__(best)
        self.mean = best
        self.stdev = sigma
        self.negative_allowed = negative_allowed

    def from_unit_interval(self, u):
        '''
        Function to convert value u in range (0,1], will convert to a value to
        be used by Batman
        '''
        if u > 1 or u < 0:
            raise ValueError('u must satisfy 0 < u < 1')

        val = self.mean + self.stdev * np.sqrt(2) * erfinv(2 * u - 1)
        if self.negative_allowed:
            return val
        return abs(val)
