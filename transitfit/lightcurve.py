'''

LightCurve objects for TransitFit

'''

import numpy as np
from .detrending_funcs import NthOrderDetrendingFunction
from .detrender import DetrendingFunction
from copy import deepcopy
import csv
from scipy.optimize import curve_fit
import os

class LightCurve:
    '''
    The transit data which we are trying to fit.

    The LightCurve is designed
    to simplify dealing with detrending etc across multiple data sets. It
    allows us to clearly keep values pointed at specific data sets.

    Parameters
    ----------
    times : array_like, shape (X, )
        The times of observation
    flux : array_like, shape (X, )
        The fluxes
    errors : array_like, shape (X, )
        The absolute uncertainty on the fluxes
    telescope_idx : int, optional
        The telescope index associated with this light curve
    filter_idx : int, optional
        The filter index associated with this light curve
    epoch_idx : int, optional
        The epoch index associated with this light curve
    curve_labels : array_like, shape (X, ), optional
        Used to identify if different data points come from different
        observations. Useful when combining curves if you want to undo that!
    telescope_array : array_like, shape (X, ), optional
        Array of the telescope indices for each data point. Useful when dealing
        with combined curves.
    filter_array : array_like, shape (X, ), optional
        Array of the filter indices for each data point. Useful when dealing
        with combined curves.
    epoch_array : array_like, shape (X, ), optional
        Array of the epoch indices for each data point. Useful when dealing
        with combined curves.
    '''
    def __init__(self, times, flux, errors, telescope_idx=None,
                 filter_idx=None, epoch_idx=None, curve_labels=None,
                 telescope_array=None, filter_array=None, epoch_array=None):


        times = np.array(times)
        flux = np.array(flux)
        errors = np.array(errors)

        if not times.ndim == 1:
            raise ValueError('Times, flux, and errors have {} dimensions. They should only have 1!'.format(times.ndims))

        self.times = times
        self.flux = flux
        self.errors = errors

        ###############################################################
        # Identifying indices - for unqiuely identifying a light curve:

        # These are quick-access references for non-combined curves
        self.telescope_idx = telescope_idx
        self.filter_idx = filter_idx
        self.epoch_idx = epoch_idx

        # These will be arrays which allow individual light curves to be
        # separated if this is a combined curve. Will be used in outputs too.
        if telescope_array is None:
            if telescope_idx is None:
                self._telescope_array = np.full(times.shape, None)
            else:
                self._telescope_array = np.ones(times.shape) * telescope_idx
        else:
            telescope_array = np.array(telescope_array)
            if not telescope_array.shape == times.shape:
                raise ValueError('telescope_array has shape {}, but requires shape {}'.format(telescope_array.shape, times.shape))
            self._telescope_array = telescope_array

        if filter_array is None:
            if filter_idx is None:
                self._filter_array = np.full(times.shape, None)
            else:
                self._filter_array = np.ones(times.shape) * filter_idx
        else:
            filter_array = np.array(filter_array)
            if not filter_array.shape == times.shape:
                raise ValueError('filter_array has shape {}, but requires shape {}'.format(filter_array.shape, times.shape))
            self._filter_array = filter_array

        if epoch_array is None:
            if epoch_idx is None:
                self._epoch_array = np.full(times.shape, None)
            else:
                self._epoch_array = np.ones(times.shape) * epoch_idx
        else:
            epoch_array = np.array(epoch_array)
            if not epoch_array.shape == times.shape:
                raise ValueError('epoch_array has shape {}, but requires shape {}'.format(epoch_array.shape, times.shape))
            self._epoch_array = epoch_array

        # These are used to label the data so that combined lightcurves
        # can be quickly untangled.
        if curve_labels is None:
            self.curve_labels = np.zeros(times.shape)
        else:
            curve_labels = np.array(curve_labels)
            if not curve_labels.shape == times.shape:
                raise ValueError('curve_labels has shape {}, but requires shape {}'.format(curve_labels.shape, times.shape))
            self.curve_labels = curve_labels

        self.detrend = False
        self.normalise = False

    def set_detrending(self, method, order=None, function=None,
                       method_idx=None):
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
        elif method.lower() == 'off':
            self.detrending_method_idx = method_idx
            self.detrend = False
            return
        else:
            raise ValueError('Unrecognised method {}'.format(method))

        self.detrending_method_idx = method_idx
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
            ``0.5/f_max <= c_n <= 1.5/f_min``
        as the default range, where f_median is the median flux value.
        '''
        self.normalise = True

        return self.estimate_normalisation_limits()

    def estimate_normalisation_limits(self):
        '''
        Estimates the range for fitting a normalisation constant, and also
        finds a reasonable first guess.

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
            ``0.5/f_max <= c_n <= 1.5/f_min``
        as the default range.
        '''
        median_factor = 1/np.median(self.flux)

        # updated v0.10.0 - gives a narrower prior
        low_factor = 0.5/np.max(self.flux)
        high_factor = 1.5/np.min(self.flux)

        return median_factor, low_factor, high_factor

    def detrend_flux(self, d, norm=1, force_normalise=False):
        '''
        When given a normalisation constant and some detrending parameters,
        will return the detrended flux and errors

        Parameters
        ----------
        d : array_like, shape(n_detrending_params,)
            Array of the detrending parameters to use
        norm : float, optional
            The normalisation constant to use. Default is 1
        force_normalise : bool, optional
            Override to allow normalisation to be forced if the LightCurve does
            not have normalisation initialised. Default is False.

        Returns
        -------
        detrend_flux : array_like, shape(num_times)
            The detrended flux
        detrended_errors : array_like, shape(num_times)
            The errors on the detrended flux
        '''

        if self.detrend and d is not None:
            detrended_flux = self.detrending_function(self, *d)
        else:
            detrended_flux = deepcopy(self.flux)
        detrended_errors = deepcopy(self.errors)

        if self.normalise or force_normalise:
            detrended_flux *= norm
            detrended_errors *= norm

        return detrended_flux, detrended_errors

    def create_detrended_LightCurve(self, d, norm):
        '''
        Creates a detrended LightCurve using detrend_flux() and returns it
        '''
        detrended_flux, detrended_errors = self.detrend_flux(d, norm)

        return LightCurve(self.times, detrended_flux, detrended_errors,
                          self.telescope_idx, self.filter_idx,
                          self.epoch_idx, self.curve_labels,
                          self._telescope_array, self._filter_array,
                          self._epoch_array)

    def fold(self, t0, period, base_t0=None):
        '''
        Folds the LightCurve so that all times are between t0 - period/2 and
        t0 + period/2.

        If base_t0 is provided, this will ensure that the folded lightcurve is
        centred on base_t0. This is to allow for ttv mode where the differences
        between the retrieved t0 for each epoch must be accounted for.

        returns a new LightCurve
        '''
        if base_t0 is None:
            base_t0 = t0

        #ttv_term = t0 - ((t0 - (base_t0+period/2))//period) * period - base_t0
        #ttv_term = self.times - ((self.times - (t0 + period/2))//period) * period - base_t0

        #times = self.times - ((self.times - (t0 + period/2))//period) * period - ttv_term
        phase = self.get_phases(t0, period)

        n = (self.times - (t0 - 0.5*period))//period

        times = base_t0 + period * (phase - 0.5)

        return LightCurve(times, self.flux, self.errors, self.telescope_idx,
                          self.filter_idx, self.epoch_idx, self.curve_labels,
                          self._telescope_array, self._filter_array,
                          self._epoch_array)

    def combine(self, lightcurves, telescope_idx=None, filter_idx=None,
                epoch_idx=None):
        '''
        Combines the LightCurve with a list of other lightcurves which are passed to it
        and returns a new lightcurve containing all the data of the input
        curves. Note that you should probably only do this with lightcurves
        which have already been detrended, otherwise you might struggle
        to detrend the combined one.
        '''
        times = self.times
        flux = self.flux
        errors = self.errors

        labels = self.curve_labels
        telescope_labels = self._telescope_array
        filter_labels = self._filter_array
        epoch_labels = self._epoch_array

        starting_label = labels.max() + 1
        for i, lightcurve in enumerate(lightcurves):
            times = np.hstack((times, lightcurve.times))
            flux = np.hstack((flux, lightcurve.flux))
            errors = np.hstack((errors, lightcurve.errors))
            labels = np.hstack((labels, np.ones(lightcurve.times.shape) * (starting_label + i)))
            telescope_labels = np.hstack((telescope_labels, lightcurve._telescope_array))
            filter_labels = np.hstack((filter_labels, lightcurve._filter_array))
            epoch_labels = np.hstack((epoch_labels, lightcurve._epoch_array))

        return LightCurve(times, flux, errors, telescope_idx, filter_idx,
                          epoch_idx, labels, telescope_labels, filter_labels,
                          epoch_labels)

    def decombine(self):
        '''
        Splits a lightcurve according to its curve labels.

        Inverse of combine
        '''
        lightcurves = []

        for label in np.unique(self.curve_labels):
            mask = self.curve_labels == label
            times = self.times[mask]
            flux = self.flux[mask]
            errors = self.errors[mask]
            telescope_array = self._telescope_array[mask]
            filter_array = self._filter_array[mask]
            epoch_array = self._epoch_array[mask]

            lightcurves.append(LightCurve(times, flux, errors,
                                          telescope_array[0], filter_array[0],
                                          epoch_array[0], None, telescope_array,
                                          filter_array, epoch_array))

        return lightcurves

    def split(self, t0, P, t14, cutoff=0.25, window=None):
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
            The estimated period of the planet in days
        t14 : float, optional
            The approximate transit duration in minutes.
        cutoff : float, optional
            If there are no data within t14 * cutoff of t0, a period will be
            discarded. Default is 0.25
        window : float, optional
            If provided, data outside of the range [t0 ± (0.5 * t14) * window]
            will be discarded.

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
        n_periods = int(n_periods) +1

        # Make sure that t0 will fall in the first new epoch
        t0 = t0 - ((t0 - self.times.min())//P * P)

        # Convert t14 to days:
        t14 /= 60 * 24

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
                if t < t0 + (2 * j + 1) * P/2 and t > t0 - P/2:
                    t_new[j].append(t)
                    f_new[j].append(f)
                    err_new[j].append(err)
                    break

        # Convert to np arrays
        for i in range(len(t_new)):
            t_new[i] = np.array(t_new[i])
            f_new[i] = np.array(f_new[i])
            err_new[i] = np.array(err_new[i])

        # Now we have all the data for the epochs, make the new LightCurves
        lightcurves = []
        for i in range(n_periods):
            if not len(t_new[i]) == 0:
                # Make sure we don't produce empty curves

                # Now a bunch more checks. Need to ensure there is data around
                # the expected t0 (ie we have a transit!). We keep the transit
                # if there is data within t14*cutoff of the predicted t0
                # (found by folding each epoch)
                if np.any(abs((t_new[i]- ((t_new[i] - t0)//P) * P) - t0) <= t14*cutoff):

                    # Now we cut out the window:
                    if window is None:
                        mask = np.ones(len(t_new[i]), bool)
                    else:
                        mask = abs((t_new[i] - ((t_new[i] - (t0 - P/2))//P) * P) - t0) <= (window * t14 * 0.5)

                    new_curve = LightCurve(t_new[i][mask], f_new[i][mask], err_new[i][mask])

                    lightcurves.append(new_curve)
                else:
                    print('Light curve {} discarded'.format(i))

        return lightcurves

    def save(self, filepath):
        '''
        Saves the LightCurve to a .csv file
        '''
        if not filepath[-4:] == '.csv':
            filepath += '.csv'

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

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

    def get_phases(self, t0, P):
        '''
        Converts times into phase given t0 and P values, with t0 at phase=0.5
        '''
        n = (self.times - (t0 - 0.5*P))//P

        return (self.times-t0)/P - n + 0.5

    def bin(self, cadence, residuals=None):
        '''
        Bins the light curve to a given observation cadence

        Parameters
        ----------
        cadence : float
            The observation cadence to bin to in the units of self.times
        residuals : array_like, shape (n_times,)
            If provided, will also bin any model residuals

        Returns
        -------
        time : array_like
            The time centers of each bin
        flux : array_like
            The flux in each bin, calculated as the weighted mean of all flux
            values in the bin
        err : array_like
            The error on the flux of each bin
        '''
        if residuals is not None:
            residuals = np.array(residuals)

        # Calculate the number of bins we need, and the size of each bin
        obs_length = self.times.max() - self.times.min()

        n_bins = int((obs_length)/cadence)
        bin_size = obs_length / n_bins

        times = np.full(n_bins, None)
        flux = np.full(n_bins, None)
        err = np.full(n_bins, None)
        binned_residuals = np.full(n_bins, None)

        points_in_bin = np.zeros(n_bins)

        # Now we can bin the data
        for i in range(n_bins):
            bin_upper = (i+1) * bin_size + self.times.min()
            bin_lower = i * bin_size + self.times.min()
            bin_time = (bin_upper + bin_lower)/2

            mask = (bin_lower <= self.times) * (self.times < bin_upper)

            bin_flux = self.flux[mask]

            bin_err = self.errors[mask]

            if len(bin_flux > 0):
                if residuals is not None:
                    bin_residuals = residuals[mask]
                    binned_residuals[i] = np.average(bin_residuals, weights=bin_err)

                times[i] = bin_time
                flux[i] = np.average(bin_flux, weights=bin_err)
                err[i] = 1/np.sqrt(np.sum(1/(bin_err**2)))

            points_in_bin[i] = np.sum(mask)

        return_mask = (points_in_bin > 0)

        if residuals is None:
            return times[return_mask], flux[return_mask], err[return_mask]
        else:
            return times[return_mask], flux[return_mask], err[return_mask], binned_residuals[return_mask]

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
