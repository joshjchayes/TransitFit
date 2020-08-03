'''

LightCurve objects for TransitFit

'''

import numpy as np
from .detrending_funcs import NthOrderDetrendingFunction
from .detrender import DetrendingFunction
from copy import deepcopy
import csv

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

    def estimate_normalisation_limits(self):
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
            ``1/f_min <= c_n <= 1/f_max``
        as the default range.
        '''
        median_factor = 1/np.median(self.flux)

        # updated v0.10.0 - gives a narrower prior
        low_factor = 1/np.max(self.flux)
        high_factor = 1/np.min(self.flux)

        return median_factor, low_factor, high_factor


    def detrend_flux(self, d, norm=1, use_full_times=False):
        '''
        When given a normalisation constant and some detrending parameters,
        will return the detrended flux and errors

        Parameters
        ----------
        d : array_like, shape(n_detrending_params,)
            Array of the detrending parameters to use
        norm : float, optional
            The normalisation constant to use. Default is 1
        use_full_times : bool, optional
            If True, will use the full BJD value of the times. If False, will
            subtract the integer part of self.times[0] from all the time values
            before passing to the detrending function. Default is False

        Returns
        -------
        detrend_flux : array_like, shape(num_times)
            The detrended flux
        detrended_errors : array_like, shape(num_times)
            The errors on the detrended flux
        '''
        detrended_flux = deepcopy(self.flux)
        detrended_errors = deepcopy(self.errors)



        if self.detrend and d is not None:
            if use_full_times:
                subtract_val = 0
            else:
                # Since times are in BJD, the detrending function results are
                # MASSIVE. We detrend using only the fractional part of the
                # times as this significantly reduces the range of each of the
                # detrending coefficients
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

    def split(self, t0, P):
        '''
        Splits the LightCurve into multiple LightCurves containing a single
        transit. This is useful for dealing with multi-epoch observations which
        contain TTVs, or have long term periodic trends, since single-epoch
        observation trends can be approximated with polynomial fits.

        Parameters
        ----------
        t0 : float
            The centre of a transit
        P : float
            The estimated period of the planet

        Returns
        -------
        lightcurves : list
            A list of LightCurve objects

        Notes
        -----
        Will reset the telescope, filter and epoch indices to None.
        '''
        # Work out how many periods are contained in the current LightCurve
        n_periods = np.ceil((self.times.max() - self.times.min())/P)
        n_periods = int(n_periods)

        # Make sure that t0 will fall in the first new epoch
        t0 = t0 - ((t0 - self.times.min())//P * P)


        # Work out the times, flux, and errors for each epoch
        t_new = [[] for i in range(n_periods)]
        f_new = [[] for i in range(n_periods)]
        err_new = [[] for i in range(n_periods)]

        # Loop through each data point and assign to the correct epoch
        for i in range(len(self.times)):
            t = self.times[i]
            f = self.flux[i]
            err = self.errors[i]

            for j in range(n_periods):
                if t < t0 + (2 * j + 1) * P/2:
                    t_new[j].append(t)
                    f_new[j].append(f)
                    err_new[j].append(err)
                    break

        # Now we have all the data for the epochs, make the new LightCurves
        lightcurves = []
        for i in range(n_periods):
            if not t_new[i] == []:
                # Make sure we don't produce empty curves
                new_curve = LightCurve(t_new[i], f_new[i], err_new[i])
                lightcurves.append(new_curve)

        return lightcurves

    def save(self, filepath):
        '''
        Saves the LightCurve to a .csv file
        '''
        if not filepath[-4:] == '.csv':
            filepath += '.csv'

        write_dict = []
        for j, tj in enumerate(self.times):
            write_dict.append({'Time' : tj,
                               'Flux' : self.flux[j],
                               'Flux_err' : self.errors[j]})

        with open(filepath, 'w') as f:
            columns = ['Time', 'Flux', 'Flux_err']
            writer = csv.DictWriter(f, columns)
            writer.writeheader()
            writer.writerows(write_dict)

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
