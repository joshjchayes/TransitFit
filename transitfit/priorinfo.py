'''
PriorInfo objects

Object to handle and deal with prior info for retrieval
'''

import numpy as np

class PriorInfo:
    def __init__(self, P=[0.01, 1000], a=[0.01, 500], inc=[85,90], t0=[-100,100],
                 rp=[0.01,0.6]):
        '''
        This is an object to handle priors.

        The parameters we are fitting are
        P - period
        a - semi-major axis (in stellar radii)
        inc - inclination (in degrees)
        t0 - time of inferior conjunction
        rp - planet radius (in stellar radii)
        '''
        self.priors = {}
        self.set_priors(P, a, inc, t0, rp)

    def set_priors(self, P=None, a=None, inc=None, t0=None, rp=None):
        '''
        Sets the priors to the vaules specified

        If provided, each prior should be a 2-tuple or list of 2 numbers,
        given in the order [lower, upper]
        '''
        if P is not None:
            self._verify_prior_format(P, 'P')
            self.priors['P'] = P
        if a is not None:
            self._verify_prior_format(a, 'a')
            self.priors['a'] = a
        if inc is not None:
            self._verify_prior_format(inc, 'inc')
            self.priors['inc'] = inc
        if t0 is not None:
            self._verify_prior_format(t0, 't0')
            self.priors['t0'] = t0
        if rp is not None:
            self._verify_prior_format(rp, 'rp')
            self.priors['rp'] = rp


    def _verify_prior_format(self, prior, prior_name):
        '''
        Checks that a value provided for a prior is either a tuple or list of
        the form [lower, upper].

        Also verifies that values are physically allowed
        '''
        if not len(prior) == 2:
            raise ValueError("You must supply both the lower and upper bound for {}".format(prior_name))
        if not prior[0] < prior[1]:
            raise ValueError("Lower bound {} >= upper bound {} for {}".format(prior[0], prior[1], prior_name))

        if prior[0] <= 0 and not prior_name=='t0':
            raise ValueError("Lower bound cannot be <0 for {}".format(prior_name))

        if prior_name == 'inc':
            if prior[0] > 90 or prior[1]>90:
                raise ValueError('Inclination cannot be >90 degrees')

    def _value_from_unit_interval(self, x, param):
        '''
        when given a value x in range (0,1], will convert to a value to be used
        by Batman
        '''

        return x * (self.priors[param][1] - self.priors[param][0]) + self.priors[param][0]
