'''
Module to calculate the likelihood of a set of parameters

'''

import numpy as np
import batman


class LikelihoodCalculator:
    def __init__(self, times, depths, errors):
        '''
        This class can be used to quickly calculate the likelihood of a set
        of parameters to fit a given data set.

        self.batman_params is a batman.TransitParams object which we 
        continually update to calculate the model transit curve.

        '''
        self.times = times
        self.depths = depths
        self.errors = errors

        self.batman_params = batman.TransitParams()

    def find_likelihood(self, t0, per, rp, a, inc, ecc, w, limb_dark, u):
        '''
        Calculates the likelihood of a set of parameters matching the given
        model
        '''
        # Update the parameters to the ones we are interested in
        self.update_params(t0, per, rp, a, inc, ecc, w, limb_dark, u)

        # Calculate the depths
        model = batman.TransitModel(self.batman_params, self.times)
        model_depths = model.light_curve(self.batman_params)

        # Work out the chi2 of the fit
        # Assuming that the data is rescaled to a baseline flux of 1.
        chi2 = sum((model_depths - self.depths)**2 / self.errors**2)

        # The likelihood is just -1*chi2
        return - chi2

    def update_params(self, t0=None, per=None, rp=None, a=None, inc=None,
                      ecc=None, w=None, limb_dark=None, u=None):
        '''
        Updates self.params with values given
        '''
        if t0 is not None:
            self.batman_params.t0 = t0
        if per is not None:
            self.batman_params.per = per
        if rp is not None:
            self.batman_params.rp = rp
        if a is not None:
            self.batman_params.a = a
        if inc is not None:
            self.batman_params.inc = inc
        if ecc is not None:
            self.batman_params.ecc = ecc
        if w is not None:
            self.batman_params.w = w
        if limb_dark is not None:
            self.batman_params.limb_dark = limb_dark
        if u is not None:
            self.batman_params.u = u
