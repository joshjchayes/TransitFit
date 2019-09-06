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
        # We need to check that

        if len(np.shape(times[0])) == 0:
            # Reshape to add extra dimension
            times = [times]
            depths = [depths]
            errors = [errors]

        '''
        if not times.shape == depths.shape == errors.shape:
            raise ValueError('Shape of times ({}), depths({}), and errors({}) do not match'.format(times.shape, depths.shape, errors.shape))

        if times.ndim > 2:
            raise ValueError('Too many dimensions for arrays, must be 1 or 2, you gave {}'.format(times.ndim))
        '''


        self.num_light_curves = len(times)

        self.times = times
        self.depths = depths
        self.errors = errors

        # We need to make a separate TransitParams for each light curve.
        self.batman_params = [batman.TransitParams() for i in range(self.num_light_curves)]

    def find_likelihood(self, t0, per, rp, a, inc, ecc, w, limb_dark, u):
        '''
        Calculates the likelihood of a set of parameters matching the given
        model

        rp and t0 should be array_like with shape (M,) if times, depths and
        errors all have shape (M,N)
        '''
        try:
            t0 = self._validate_variant_parameter(t0)
        except:
            print('Invalid t0')
            raise
        try:
            rp = self._validate_variant_parameter(rp)
        except:
            print('Invalid rp')
            raise


        all_chi2 = []

        for i in range(self.num_light_curves):
            # Update the parameters to the ones we are interested in
            self.update_params(i, t0[i], per, rp[i], a, inc, ecc, w, limb_dark, u)

            # Calculate the transits and the chi2 values
            model = batman.TransitModel(self.batman_params[i], self.times[i])
            model_depths = model.light_curve(self.batman_params[i])

            # Work out the chi2 of the fit
            # Assuming that the data is rescaled to a baseline flux of 1.
            all_chi2.append(sum((model_depths - self.depths[i])**2 / self.errors[i]**2))

        # The likelihood is just -1*chi2
        return - sum(all_chi2)

    def update_params(self, light_curve_number, t0=None, per=None, rp=None,
                      a=None, inc=None, ecc=None, w=None, limb_dark=None,
                      u=None):
        '''
        Updates self.params with values given
        '''
        if t0 is not None:
            self.batman_params[light_curve_number].t0 = t0
        if per is not None:
            self.batman_params[light_curve_number].per = per
        if rp is not None:
            self.batman_params[light_curve_number].rp = rp
        if a is not None:
            self.batman_params[light_curve_number].a = a
        if inc is not None:
            self.batman_params[light_curve_number].inc = inc
        if ecc is not None:
            self.batman_params[light_curve_number].ecc = ecc
        if w is not None:
            self.batman_params[light_curve_number].w = w
        if limb_dark is not None:
            self.batman_params[light_curve_number].limb_dark = limb_dark
        if u is not None:
            self.batman_params[light_curve_number].u = u

    def _validate_variant_parameter(self, p):
        '''
        Checks that a parameter which is either epoch or wavelength variant
        is in a format which can be used, and returns it in a usable (iterable)
        format
        '''
        if type(p) == int or type(p) == float:
            p = [p]

        p = np.asarray(p)

        if not p.shape[0] == self.num_light_curves:
            print(p)
            raise ValueError('Incorrect number {} of parameters provided for fitting {} lightcuves.'.format(p.shape[0], self.num_light_curves))

        return p
