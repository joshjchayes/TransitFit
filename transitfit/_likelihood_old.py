'''
Module to calculate the likelihood of a set of parameters


This is a WIP to couple the rp and t0 for same wavelength and epoch
respectively
'''

import numpy as np
import batman
from copy import deepcopy



class LikelihoodCalculator:
    def __init__(self, lightcurves, priorinfo):
        '''
        Object to quickly calculate the likelihood of a set of parameters to
        fit a given set of light curves.

        Parameters
        ----------
        lightcurves : array_like, shape (n_telescopes, n_filters, n_epochs)
            An array of LightCurves. If no data exists for a point in the array
            then the entry should be `None`.
        priorinfo : PriorInfo
            The PriorInfo object for retrieval
        '''
        lightcurves = deepcopy(lightcurves)
        self.lightcurves = np.array(lightcurves, dtype=object)

        self.n_telescopes = self.lightcurves.shape[0]
        self.n_filters = self.lightcurves.shape[1]
        self.n_epochs = self.lightcurves.shape[2]

        self.num_light_curves = len(np.where(self.lightcurves.flatten() != None)[0])

        self.priors = priorinfo

        # We need to make a separate TransitParams and TransitModels for each
        # light curve.

        # Initialise them:
        self.batman_params = np.full((self.n_telescopes, self.n_filters, self.n_epochs), None, object)
        self.batman_models = np.full((self.n_telescopes, self.n_filters, self.n_epochs), None, object)

        for i in np.ndindex(self.lightcurves.shape):
            if self.lightcurves[i] is not None:
                # Set up the params
                self.batman_params[i] = batman.TransitParams()

                # Set up the TransitModels
                # Make some realistic parameters to setup the models with
                default_params = batman.TransitParams()
                if self.priors.fit_ttv:
                    default_params.t0 = priorinfo.priors['t0'][0,0,0].default_value
                else:
                    default_params.t0 = priorinfo.priors['t0'].default_value
                default_params.per = priorinfo.priors['P'].default_value
                default_params.rp = priorinfo.priors['rp'][i[1]].default_value
                default_params.a = priorinfo.priors['a'].default_value
                default_params.inc = priorinfo.priors['inc'].default_value
                default_params.ecc = priorinfo.priors['ecc'].default_value
                default_params.w = priorinfo.priors['w'].default_value
                default_params.limb_dark = priorinfo.limb_dark
                u = [priorinfo.priors[uX][i[1]].default_value for uX in priorinfo.limb_dark_coeffs]
                default_params.u = u

                # Now make the models
                self.batman_models[i] = batman.TransitModel(default_params, self.lightcurves[i].times)


    def find_likelihood(self, t0, per, rp, a, inc, ecc, w, limb_dark, q,
                        norm, d=None, use_full_times=False):
        '''
        Calculates the ln likelihood of a set of parameters matching the given
        model

        Parameters
        ----------
        t0 : float
            t0 value
        per : float
            The period, in the same units as t0
        rp : array_like, shape (n_filters,)
            The planet radii for each filter
        a : float
            The semimajor axis
        inc : float
            Orbital inclination in degrees
        ecc : float
            Orbital inclination
        w : float
            The angle of periastron
        limb_dark : str
            The limb darkening model to use
        q : array_like, shape (n_filters, n_ld_coeffs)
            The limb darkening coefficients, as Kipping parameters
        norm : array_like, shape(n_telescopes, n_filters, n_epochs)
            The normalisation constants
        d : array_like, shape(n_telescopes, n_filters, n_epochs)
            Each entry should be a list containing all the detrending
            coefficients to trial.
        use_full_times : bool, optional
            If True, will use the full BJD value of the times. If False, will
            subtract the integer part of self.times[0] from all the time values
            before passing to the detrending function. Default is False

        Returns
        -------
        lnlike : float
            The ln likelihood of the parameter set
        '''

        if not len(rp) == self.n_filters:
            raise ValueError('You supplied {} rp values, not {} as expected'.format(len(rp), self.n_filters))

        all_chi2 = []
        n_data_points = 0

        for i in np.ndindex(self.lightcurves.shape):
            telescope_idx = i[0]
            filter_idx = i[1]
            epoch_idx = i[2]

            if self.batman_params[i] is not None:
                # update the parameters to the testing ones
                u = self.priors.ld_handler.convert_qtou(*q[filter_idx])

                if self.priors.fit_ttv:
                    self.update_params(telescope_idx, filter_idx, epoch_idx, t0[telescope_idx, filter_idx, epoch_idx], per, rp[filter_idx], a, inc, ecc, w, limb_dark, u)
                else:
                    self.update_params(telescope_idx, filter_idx, epoch_idx, t0, per, rp[filter_idx], a, inc, ecc, w, limb_dark, u)

                # Calculate the model transits
                model = self.batman_models[i]
                model_flux = model.light_curve(self.batman_params[i])

                # Get the detrended/normalised flux from the LightCurves
                if d is None:
                    comparison_flux, err = self.lightcurves[i].detrend_flux(d, norm[i], use_full_times)
                else:
                    comparison_flux, err = self.lightcurves[i].detrend_flux(d[i], norm[i], use_full_times)

                # Work out the chi2
                chi2 = sum((model_flux - comparison_flux)**2 / err**2)

                # Check to make sure that there is actually a transit in the model
                # otherwise we impose a large penalty to the chi2 value
                # This avoids a situation where the detrending values try
                # to completely flatten the light curves, which is wrong!
                if np.isclose(model_flux, 1).all():
                    chi2 += 10000000

                all_chi2.append(chi2)

        return - sum(all_chi2)


    def update_params(self, telescope_idx, filter_idx, epoch_index, t0=None,
                      per=None, rp=None, a=None, inc=None, ecc=None, w=None,
                      limb_dark=None, u=None):
        '''
        Updates self.params with values given
        '''
        if t0 is not None:
            self.batman_params[telescope_idx, filter_idx, epoch_index].t0 = t0
        if per is not None:
            self.batman_params[telescope_idx, filter_idx, epoch_index].per = per
        if rp is not None:
            self.batman_params[telescope_idx, filter_idx, epoch_index].rp = rp
        if a is not None:
            self.batman_params[telescope_idx, filter_idx, epoch_index].a = a
        if inc is not None:
            self.batman_params[telescope_idx, filter_idx, epoch_index].inc = inc
        if ecc is not None:
            self.batman_params[telescope_idx, filter_idx, epoch_index].ecc = ecc
        if w is not None:
            self.batman_params[telescope_idx, filter_idx, epoch_index].w = w
        if limb_dark is not None:
            self.batman_params[telescope_idx, filter_idx, epoch_index].limb_dark = limb_dark
        if u is not None:
            self.batman_params[telescope_idx, filter_idx, epoch_index].u = u

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
            raise ValueError('Incorrect number {} of parameters provided for fitting {} lightcurves.'.format(p.shape[0], self.num_light_curves))

        return p
