'''
Module to calculate the likelihood of a set of parameters


This is a WIP to couple the rp and t0 for same wavelength and epoch
respectively
'''

import numpy as np
import batman
from copy import deepcopy


class LikelihoodCalculator:
    def __init__(self, times, depths, errors):
        '''
        This class can be used to quickly calculate the likelihood of a set
        of parameters to fit a given data set.

        self.batman_params is a batman.TransitParams object which we
        continually update to calculate the model transit curve.

        each of times, depths, errors needs to be an MxN array where M is the
        number of different wavelengths being used, and N is is time (each
        transit). This allows us to couple light curves together to help
        work out rp and t0 in a coupled manner.

        If no light curve exists for a point in the array, the entry should be
        None
        '''
        times = np.array(times, dtype=object)
        depths = np.array(depths, dtype=object)
        errors = np.array(errors, dtype=object)

        self.num_wavelengths = times.shape[0]
        self.num_times = times.shape[1]

        self.num_light_curves = len(np.where(times == None)[0])

        self.times = times
        self.depths = depths
        self.errors = errors

        # We need to make a separate TransitParams for each light curve.
        # Initialse it:
        self.batman_params = np.array([[None for i in range(self.num_times)] for j in range(self.num_wavelengths)])

        for i in range(self.num_wavelengths):
            for j in range(self.num_times):
                if self.times[i][j] is not None:
                    self.batman_params[i][j] = batman.TransitParams()

    def find_likelihood(self, t0, per, rp, a, inc, ecc, w, limb_dark, u,
                        norm, detrend_function=None, d=None):
                        #TODO: add in normalisation
        '''
        Calculates the likelihood of a set of parameters matching the given
        model

        t0 should be array with length self.num_times

        rp should be array with lengths self.num_wavelengths
        '''
        if not len(t0) == self.num_times:
            raise ValueError('You supplied {} t0 values, not {} as expected'.format(len(t0), self.num_times))

        if not len(rp) == self.num_wavelengths:
            raise ValueError('You supplied {} rp values, not {} as expected'.format(len(rp), self.num_wavelengths))

        #print('----')

        all_chi2 = []
        for i in range(self.num_wavelengths):
            for j in range(self.num_times):
                if self.batman_params[i,j] is not None:
                    # Update the parameters to the ones we are interested in
                    self.update_params(i, j, t0[j], per, rp[i], a, inc, ecc, w, limb_dark, u[i])

                    # Calculate the transits and the chi2 values
                    model = batman.TransitModel(self.batman_params[i,j], self.times[i][j])
                    model_depths = model.light_curve(self.batman_params[i,j])

                    comparison_depths = deepcopy(self.depths[i][j])
                    #print(comparison_depths.shape)
                    comparison_depths *= norm[i, j]
                    #print('norm:', norm[i, j])
                    #print('comparison_depths mean:', np.mean(comparison_depths))

                    # If detrending is happening, it does on here!
                    if detrend_function is not None:
                        if d is None:
                            raise TypeError('Detrend function given but d is None!')
                        #print(d[:,i,j])

                        # Because we are taking times in BJD, the detrend
                        # function results are MASSIVE. We will detrend using
                        # only the decimal part of the times as this
                        # significantly reduces the range of each of the
                        # detrending coefficients
                        subtract_val = np.floor(self.times[i][j][0])

                        detrend_values = detrend_function(self.times[i][j] - subtract_val, *d[:,i,j])

                        #print('Detrending coeffs:', *d[:, i, j])
                        #print('mean detrend value', detrend_values.mean())

                        comparison_depths -= detrend_values
                        #print('Mean depths:', np.mean(comparison_depths))
                    # Work out the chi2 of the fit
                    # Assuming that the data is rescaled to a baseline flux of 1.
                    chi2 = sum((model_depths - comparison_depths)**2 / (self.errors[i][j] * norm[i, j])**2)
                    #print(chi2)
                    all_chi2.append(chi2)
        #print('chi2: ', all_chi2)
        # The likelihood is just -1*chi2
        return - sum(all_chi2)

    def update_params(self, wavelength_index, time_index, t0=None, per=None, rp=None,
                      a=None, inc=None, ecc=None, w=None, limb_dark=None,
                      u=None):
        '''
        Updates self.params with values given
        '''
        if t0 is not None:
            self.batman_params[wavelength_index, time_index].t0 = t0
        if per is not None:
            self.batman_params[wavelength_index, time_index].per = per
        if rp is not None:
            self.batman_params[wavelength_index, time_index].rp = rp
        if a is not None:
            self.batman_params[wavelength_index, time_index].a = a
        if inc is not None:
            self.batman_params[wavelength_index, time_index].inc = inc
        if ecc is not None:
            self.batman_params[wavelength_index, time_index].ecc = ecc
        if w is not None:
            self.batman_params[wavelength_index, time_index].w = w
        if limb_dark is not None:
            self.batman_params[wavelength_index, time_index].limb_dark = limb_dark
        if u is not None:
            self.batman_params[wavelength_index, time_index].u = u

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
