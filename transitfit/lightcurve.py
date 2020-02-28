'''

LightCurve objects for TransitFit

'''

import numpy as np
from .detrending_funcs import NthOrderDetrendingFunction
from .detrender import DetrendingFunction
from copy import deepcopy

class LightCurve:
    def __init__(self, times, flux, errors, telescope_idx=None,
                 filter_idx=None, epoch_idx=None):
        '''
        The transit data which we are trying to fit. The LightCurve is designed
        to simplify dealing with detrending etc across multiple data sets. It
        allows us to clearly keep values pointed at specific data sets
        '''

        times = np.array(times)
        flux = np.array(flux)
        errors = np.array(errors)

        if not (times.shape == flux.shape == errors.shape):
            raise ValueError('The shapes of times, flux, and errors don\'t match!')

        if not times.ndim == 1:
            raise ValueError('Times, flux, and errors have {} dimensions. They should only have 1!'.format(times.ndims))

        self.times = times
        self.flux = flux
        self.errors = errors

        self.epoch_idx = epoch_idx
        self.filter_idx = filter_idx
        self.telescope_idx = telescope_idx

        self.detrend = False
        self.normalise = False

    def set_detrending(self, method, order=None, function=None):
        '''
        Sets detrending method

        Parameters
        ----------
        method : str
            Accepted
                - 'nth order'
                - 'custom'
        order : int, optional
            The order of detrending to use if method is 'nth order'. Must be
            provided.
        function : function, optional
            If method is 'custom', this is the custom detrending function to
            use.
        '''
        if method.lower() in ['nth order', 'nthorder', 'nth_order']:
            if order is None:
                raise ValueError('You need to provide a detrending order!')

            function = NthOrderDetrendingFunction(order)
            self.detrending_function = DetrendingFunction(function)
            self.num_detrending_params = order
        elif method.lower() == 'custom':
            if function is None:
                raise ValueError('You must provide the custom detrending function')
            self.detrending_function = DetrendingFunction(function)
            self.num_detrending_params = self.detrending_function.n_required_args - 1
        else:
            raise ValueError('Unrecognised method {}'.format(method))

        self.detrend = True

    def set_normalisation(self, default_low=0.1):
        '''
        Turns on normalisation. Also estimates limits and a best initial guess.

        Parameters
        ----------
        default_low : float, optional
            The lowest value to consider as a multiplicative normalisation
            constant. Default is 0.1.

        Returns
        -------
        median_factor : float
            The initial best guess factor
        low_factor : float
            The low limit on the normalisation factor
        high_factor : float
            The high limit on the normalisation factor

        Notes
        -----
        We use
            ``1/f_median -1 <= c_n <= 1/f_median + 1``
        as the default range, where f_median is the median flux value.
        '''
        self.normalise = True

        return self.estimate_normalisation_limits(default_low)

    def estimate_normalisation_limits(self, default_low=0.1):
        '''
        Estimates the range for fitting a normalisation constant, and also
        finds a reasonable first guess.

        Parameters
        ----------
        default_low : float, optional
            The lowest value to consider as a multiplicative normalisation
            constant. Default is 0.1.

        Returns
        -------
        median_factor : float
            The initial best guess factor
        low_factor : float
            The low limit on the normalisation factor
        high_factor : float
            The high limit on the normalisation factor

        Notes
        -----
        We use
            ``1/f_median -1 <= c_n <= 1/f_median + 1``
        as the default range, where f_median is the median flux value.
        '''
        median_factor = 1/np.median(self.flux)
        if median_factor - 1 <=0:
            low_factor = default_low
        else:
            low_factor = median_factor - 1

        high_factor = median_factor + 1

        return median_factor, low_factor, high_factor


    def detrend_flux(self, d, norm):
        '''
        When given a normalisation constant and some detrending parameters,
        will return the detrended flux and errors

        Parameters
        ----------
        d : array_like, shape(num_detrending_params,), optional
            Array of the detrending parameters to use
        norm : float, optional
            The normalisation constant to use. Default is 1

        Returns
        -------
        detrend_flux : array_like, shape(num_times)
            The detrended flux
        detrended_errors : array_like, shape(num_times)
            The errors on the detrended flux
        '''
        detrended_flux = deepcopy(self.flux)
        detrended_errors = deepcopy(self.errors)

        # Since times are in BJD, the detrending function results are MASSIVE.
        # We detrend using only the fractional part of the times as this
        # significantly reduces the range of each of the detrending
        # coefficients
        if self.detrend and d is not None:
            subtract_val = np.floor(self.times[0])
            detrend_values = self.detrending_function(self.times - subtract_val, *d)

            detrended_flux -= detrend_values

        if self.normalise:
            detrended_flux *= norm
            detrended_errors *= norm

        return detrended_flux, detrended_errors
