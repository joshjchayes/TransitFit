'''
detrending.py

This functions deals with the detrending of a light curve
'''

import numpy as np


class LightCurve:
    def __init__(self, time, flux, uncertainty=None, metadata={}):
        '''
        A light curve object can be used to store data and metadata on a transit

        Parameters
        ----------
        time : array_like
            The times of each flux measurement
        flux : array_like
            The flux measurements
        uncertainty : array_like, optional
            The uncertainty on each flux measurement. If not supplied, will
            default to 0 uncertainty
        metadata : dict, optional
            Optional dictionary to store any metadata on the curve you want to.
        '''
        time = np.array(time)
        flux = np.array(flux)
        if not time.shape == flux.shape:
            raise ValueError('Time and flux shapes {} and {} do not match!'.format(time.shape, flux.shape))
        self.time = time
        self.flux = flux

        if uncertainty is None:
            self.uncertainty = np.zeros(time.shape)
        else:
            if not uncertainty.shape == flux.shape:
                raise ValueError('uncertainty shape {} does not match shape of\
                    time and flux {}!'.format(uncertainty.shape, flux.shape))
            self.uncertainty = uncertainty

        self.metadata = metadata

    def detrend(self, detrending_function, **function_args):
        '''
        Detrends the LightCurve using a given function passed function_args.
        Creates a new object rather than detrending in place.

        Parameters
        ----------
        detrending_function : function
            This should be a function which takes an array of times and some
            tuning variables and returns an array of values which should be
            subtracted from measured flux values to detrend. Should have
            signature detrending_function(times, **args)
        **function_args
            The extra arguments which should be passed to detrending_function

        Returns
        -------
        detrended_light_curve : LightCurve
            A new instance of a LightCurve with the detrending applied


        Notes
        -----
        We preserve the fractional size of the uncertainties on the detrended
        flux values
        '''
        old_flux = self.flux
        old_uncertainty = self.uncertainty

        new_flux = self.flux - detrending_function(self.time, **function_args)

        return LightCurve(self.time, new_flux, self.uncertainty, self.metadata)
