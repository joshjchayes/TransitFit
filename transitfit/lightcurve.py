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
            self.n_detrending_params = order
        elif method.lower() == 'custom':
            if function is None:
                raise ValueError('You must provide the custom detrending function')
            self.detrending_function = DetrendingFunction(function)
            self.n_detrending_params = self.detrending_function.n_required_args - 1
        else:
            raise ValueError('Unrecognised method {}'.format(method))

        self.detrend = True

    def set_normalisation(self):
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

        return self.estimate_normalisation_limits()

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

        # updated v0.10.0 - gives a narrower prior
        low_factor = 1/np.max(self.flux)
        high_factor = 1/np.min(self.flux)

        return median_factor, low_factor, high_factor


    def detrend_flux(self, d, norm=1):
        '''
        When given a normalisation constant and some detrending parameters,
        will return the detrended flux and errors

        Parameters
        ----------
        d : array_like, shape(n_detrending_params,)
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

    def create_detrended_LightCurve(self, d, norm):
        '''
        Creates a detrended LightCurve using detrend_flux() and returns it
        '''
        detrended_flux, detrended_errors = self.detrend_flux(d, norm)

        return LightCurve(self.times, detrended_flux, detrended_errors,
                          self.telescope_idx, self.filter_idx, self.epoch_idx)

    def fold(self, t0, period):
        '''
        Folds the LightCurve so that all times are between t0 - period/2 and
        t0 + period/2

        returns a new LightCurve
        '''
        times = self.times + (((t0 + period/2) - self.times)//period) * period
        return LightCurve(times, self.flux, self.errors, self.telescope_idx,
                          self.filter_idx, self.epoch_idx)

    def combine(self, *lightcurves, telescope_idx=None, filter_idx=None,
                epoch_idx=None):
        '''
        Combines the LightCurve with other lightcurves which are passed to it
        and returns a new lightcurve containing all the data of the input
        curves. Note that you should probably only do this with lightcurves
        which have already been detrended, otherwise you might struggle
        to detrend the combined one.
        '''
        times = self.times
        flux = self.flux
        errors = self.errors

        for lightcurve in lightcurves:
            times = np.hstack((times, lightcurve.times))
            flux = np.hstack((flux, lightcurve.flux))
            errors = np.hstack((errors, lightcurve.errors))

        return LightCurve(times, flux, errors, telescope_idx, filter_idx,
                          epoch_idx)

    def __eq__(self, other):
        '''
        Checks two LightCurves are the same. We are only checking the times,
        fluxes and errors.
        '''
        if not isinstance(other, LightCurve):
            return False

        if not self.times.shape == other.times.shape:
            return False
        if not np.all(self.times == other.times):
            return False

        if not self.flux.shape == other.flux.shape:
            return False
        if not np.all(self.flux == other.flux):
            return False

        if not self.errors.shape == other.errors.shape:
            return False
        if not np.all(self.errors == other.errors):
            return False

        return True
