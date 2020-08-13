'''
Module to calculate the likelihood of a set of parameters


This is a WIP to couple the rp and t0 for same wavelength and epoch
respectively
'''

import numpy as np
import batman
from copy import deepcopy
from ._paramarray import ParamArray


class LikelihoodCalculator:
    def __init__(self, lightcurves, priors):
        '''
        Object to quickly calculate the likelihood of a set of parameters to
        fit a given set of light curves.

        Parameters
        ----------
        lightcurves : array_like, shape (n_telescopes, n_filters, n_epochs)
            An array of LightCurves. If no data exists for a point in the array
            then the entry should be `None`.
        priors : PriorInfo
            The PriorInfo object for retrieval
        '''
        lightcurves = deepcopy(lightcurves)
        self.lightcurves = np.array(lightcurves, dtype=object)

        self.n_telescopes = self.lightcurves.shape[0]
        self.n_filters = self.lightcurves.shape[1]
        self.n_epochs = self.lightcurves.shape[2]

        self.num_light_curves = len(np.where(self.lightcurves.flatten() != None)[0])

        self.priors = priors

        # We need to make a separate TransitParams and TransitModels for each
        # light curve.

        # Initialise them:
        self.batman_params = ParamArray('batman_params', (self.n_telescopes, self.n_filters, self.n_epochs), True, True, True, lightcurves=self.lightcurves)
        self.batman_models = ParamArray('batman_models', (self.n_telescopes, self.n_filters, self.n_epochs), True, True, True, lightcurves=self.lightcurves)


        for i in np.ndindex(self.lightcurves.shape):
            tidx, fidx, eidx = i

            if self.lightcurves[i] is not None:
                # Set up the params
                self.batman_params.set_value(batman.TransitParams(), tidx, fidx, eidx)

                # Set up the TransitModels
                # Make some realistic parameters to setup the models with
                default_params = batman.TransitParams()
                if self.priors.fit_ttv:
                    default_params.t0 = priors.priors['t0'].default_value
                else:
                    default_params.t0 = priors.priors['t0'].default_value
                default_params.per = priors.priors['P'].default_value
                default_params.rp = priors.priors['rp'].default_value
                default_params.a = priors.priors['a'].default_value
                default_params.inc = priors.priors['inc'].default_value
                default_params.ecc = priors.priors['ecc'].default_value
                default_params.w = priors.priors['w'].default_value
                default_params.limb_dark = priors.limb_dark
                # Note that technically this is q, not u, but it doesn't matter
                # as we are only initialising here
                default_params.u = [priors.priors[qX].default_value for qX in priors.limb_dark_coeffs]

                # Now make the models
                model = batman.TransitModel(default_params, self.lightcurves[i].times)
                self.batman_models.set_value(model, tidx, fidx, eidx)


    def find_likelihood(self, params, use_full_times=False):
        '''
        Finds the likelihood of a set of parameters
        '''
        all_chi2 = []
        n_data_points = 0
        total_chi2 = 0

        for i in np.ndindex(self.lightcurves.shape):
            tidx, fidx, eidx = i

            if self.lightcurves[i] is not None:

                # GENERATE THE MODEL LIGHT CURVE

                # Convert the LDC q values to u:
                u = self.priors.ld_handler.convert_qtou(*[params[qX].get_value(*i) for qX in self.priors.limb_dark_coeffs])

                # Need to update the parameters
                self.update_params(tidx, fidx, eidx,
                                   params['t0'][i],
                                   params['P'][i],
                                   params['rp'][i],
                                   params['a'][i],
                                   params['inc'][i],
                                   params['ecc'][i],
                                   params['w'][i],
                                   self.priors.limb_dark,
                                   u)

                # Now we calculate the model transits
                model = self.batman_models[i]
                model_flux = model.light_curve(self.batman_params[i])

                # DETREND AND NORMALISE THE DATA TO COMPARE TO THE MODEL
                if self.priors.detrend:
                    # We have to work out the detrending info
                    # Get the method index
                    detrending_index = self.priors._detrend_method_index_array[i]
                    # Now get the parameter values
                    d = [params[di][i] for di in self.priors.detrending_coeffs[detrending_index]]
                else:
                    d = None

                if self.priors.normalise:
                    norm = params['norm'][i]
                else:
                    norm = 1

                detrended_flux, err = self.lightcurves[i].detrend_flux(d, norm, use_full_times)

                # Work out the chi2
                #chi2 = sum((model_flux - detrended_flux)**2 / err**2)
                total_chi2 += np.sum((model_flux - detrended_flux)**2 / err**2)

                # Check to make sure that there is actually a transit in the model
                # otherwise we impose a large penalty to the chi2 value
                # This avoids a situation where the detrending values try
                # to completely flatten the light curves, which is wrong!
                #if np.isclose(model_flux, 1).all():
                #    chi2 += 10000000

                #all_chi2.append(chi2)

        #return - sum(all_chi2)
        return - total_chi2

    def update_params(self, telescope_idx, filter_idx, epoch_index, t0=None,
                      P=None, rp=None, a=None, inc=None, ecc=None, w=None,
                      limb_dark=None, u=None):
        '''
        Updates self.params with values given
        '''
        if t0 is not None:
            self.batman_params.array[telescope_idx, filter_idx, epoch_index].t0 = t0
        if P is not None:
            self.batman_params.array[telescope_idx, filter_idx, epoch_index].per = P
        if rp is not None:
            self.batman_params.array[telescope_idx, filter_idx, epoch_index].rp = rp
        if a is not None:
            self.batman_params.array[telescope_idx, filter_idx, epoch_index].a = a
        if inc is not None:
            self.batman_params.array[telescope_idx, filter_idx, epoch_index].inc = inc
        if ecc is not None:
            self.batman_params.array[telescope_idx, filter_idx, epoch_index].ecc = ecc
        if w is not None:
            self.batman_params.array[telescope_idx, filter_idx, epoch_index].w = w
        if limb_dark is not None:
            self.batman_params.array[telescope_idx, filter_idx, epoch_index].limb_dark = limb_dark
        if u is not None:
            self.batman_params.array[telescope_idx, filter_idx, epoch_index].u = u
