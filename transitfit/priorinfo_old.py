'''
PriorInfo objects

Object to handle and deal with prior info for retrieval
'''

import numpy as np
from collections.abc import Iterable

from ._params import _Param, _UniformParam, _GaussianParam
from .detrending_funcs import NthOrderDetrendingFunction
from .detrender import DetrendingFunction
from ._limb_darkening_handler import LimbDarkeningHandler

_prior_info_defaults = {'P':1, 'a':10, 'inc':90, 'rp':0.05, 't0':0, 'ecc':0,
                        'w':90, 'limb_dark':'quadratic', 'q0':0.1, 'q1':0.3,
                        'q2':-0.1, 'q3':0, 'n_telescopes':1,  'n_filters':1,
                        'n_epochs':1, 'norm':1}


def setup_priors(P, rp, a, inc, t0, ecc, w, ld_model, n_telescopes, n_filters, n_epochs, fit_ttv=False):
    '''
    Initialises a PriorInfo object with default parameter values.
    '''
    default_dict = {'rp' : rp,
                    'P' : P,
                    'a' : a,
                    'inc' : inc,
                    't0' : t0,
                    'ecc' : ecc,
                    'w' : w,
                    'n_telescopes' : n_telescopes,
                    'n_epochs' : n_epochs,
                    'n_filters' : n_filters,
                    'limb_dark' : ld_model}

    return PriorInfo(default_dict, fit_ttv=fit_ttv, warn=False)

class PriorInfo:
    def __init__(self, default_dict, fit_ttv=False, warn=True):
        '''
        Contains all the info to be used in fitting, both the variable and
        fixed parameters.

        Generally it is inadvisable to directly initialise this object,
        and you should call the factory function setup_priors() as this does
        a lot of the formatting of inputs for you and will ensure you have
        everything you need.
        '''

        if warn:
            print('Warning: you appear to be making the PriorInfo directly rather than using setup_priors. Proceed with caution, or set up your priors using csv files. You can mute this warning by using warn=False.')

        # This is the dictionary containing all prior ranges
        self.priors = {}

        # These are lists of fitting parameters and corresponding lists of
        # the telescope, filter and epoch indices.
        self.fitting_params = []
        self._telescope_idx = []
        self._filter_idx = []
        self._epoch_idx = []

        self.n_epochs = default_dict['n_epochs']
        self.n_filters = default_dict['n_filters']
        self.n_telescopes = default_dict['n_telescopes']
        self.limb_dark = default_dict['limb_dark']

        self.fit_ttv = fit_ttv

        # Initialise limb darkening
        self.ld_handler = LimbDarkeningHandler(self.limb_dark)
        self.limb_dark_coeffs = self.ld_handler.get_required_coefficients()
        self.fit_ld = False
        self.ld_fit_method = 'off'

        for ldc in self.limb_dark_coeffs:
            # Set up the default values for the limb darkening coeffs
            self.priors[ldc] = [_Param(_prior_info_defaults[ldc]) for i in range(self.n_filters)]

        # Initialse a bit for the detrending
        self.detrend = False
        self.detrending_coeffs = []
        self.detrending_coeffs_fit_mode = []

        # Initialise normalisation things
        self.normalise=False
        self.priors['norm'] = np.full((self.n_telescopes, self.n_filters, self.n_epochs), _Param(1), object)

        # So far we have only initialised TransitFit default values. Now we
        # go through the default_dict to update to any user-supplied defaults.

        for key in default_dict.keys():
            if key in ['rp'] + self.limb_dark_coeffs:
                # Only filter-variant parameters
                # Since these are not being fitted at this point, we just set
                # them all to the same value
                self.priors[key] = np.array([_Param(default_dict[key]) for i in range(self.n_filters)])

            elif key in ['norm']:
                # light curve specific values - need all indices
                self.priors[key] = np.full((self.n_telescopes, self.n_filters, self.n_epochs), _Param(default_dict[key]), object)

            elif key in ['n_telescopes', 'n_epochs', 'n_filters', 'limb_dark']:
                # The 'do nothing' keys
                pass
            else:
                if fit_ttv and key in ['t0']:
                    # Set up t0 so it can be fitted for each lightcurve
                    # independently
                    self.priors[key] = np.full((self.n_telescopes, self.n_filters, self.n_epochs), _Param(default_dict[key]), object)
                else:
                    self.priors[key] = _Param(default_dict[key])


    def add_uniform_fit_param(self, name, low_lim, high_lim,
                              telescope_idx=None, filter_idx=None, epoch_idx=None):
        '''
        Adds a new parameter which will be fitted uniformly in the range given
        by low_lim and high_lim

        index is used if name is 'rp' or 't0' and refers to the column or
        row in the MxN array which the data we are retrieving should be in.
        Since limb darkening is wavelength specific, this should also be
        given for 'u1', 'u2', 'u3', and 'u4'.
        '''

        # Check that the parameter was initialised
        if not name in self.priors:
            raise KeyError('Unrecognised prior name {}'.format(name))

        # Deal with all the cases of the given indices - these will indicate
        # how these parameters are coupled across telescope, filter, and epoch
        if telescope_idx is None:
            if filter_idx is None:
                if epoch_idx is None:
                    self.priors[name] = _UniformParam(low_lim, high_lim)
                else:
                    self.priors[name][epoch_idx] = _UniformParam(low_lim, high_lim)
            else:
                if epoch_idx is None:
                    self.priors[name][filter_idx] = _UniformParam(low_lim, high_lim)
                else:
                    self.priors[name][filter_idx, epoch_idx] = _UniformParam(low_lim, high_lim)

        else:
            if filter_idx is None:
                if epoch_idx is None:
                    self.priors[name][telescope_idx] = _UniformParam(low_lim, high_lim)
                else:
                    self.priors[name][telescope_idx, epoch_idx] = _UniformParam(low_lim, high_lim)
            else:
                if epoch_idx is None:
                    self.priors[name][telescope_idx, filter_idx] = _UniformParam(low_lim, high_lim)
                else:
                    self.priors[name][telescope_idx, filter_idx, epoch_idx] = _UniformParam(low_lim, high_lim)

        # Store some info for later
        self.fitting_params.append(name)
        self._filter_idx.append(filter_idx)
        self._epoch_idx.append(epoch_idx)
        self._telescope_idx.append(telescope_idx)

    def add_gaussian_fit_param(self, name, mean, stdev,
                               telescope_idx=None, filter_idx=None, epoch_idx=None):
        '''
        Adds a new parameter which will be fitted using a Gaussian prior
        specified by mean and stdev

        index is used if name is 'rp' or 't0' and refers to the column or
        row in the MxN array which the data we are retrieving should be in.
        Since limb darkening is wavelength specific, this should also be
        given for 'u1', 'u2', 'u3', and 'u4'.
        '''

        # Check that the parameter was initialised
        # Check that the parameter was initialised
        if not name in self.priors:
            raise KeyError('Unrecognised prior name {}'.format(name))

        # Deal with all the cases of the given indices - these will indicate
        # how these parameters are coupled across telescope, filter, and epoch
        if telescope_idx is None:
            if filter_idx is None:
                if epoch_idx is None:
                    self.priors[name] = _GaussianParam(mean, stdev)
                else:
                    self.priors[name][epoch_idx] = _GaussianParam(mean, stdev)
            else:
                if epoch_idx is None:
                    self.priors[name][filter_idx] = _GaussianParam(mean, stdev)
                else:
                    self.priors[name][filter_idx, epoch_idx] = _GaussianParam(mean, stdev)

        else:
            if filter_idx is None:
                if epoch_idx is None:
                    self.priors[name][telescope_idx] = _GaussianParam(mean, stdev)
                else:
                    self.priors[name][telescope_idx, epoch_idx] = _GaussianParam(mean, stdev)
            else:
                if epoch_idx is None:
                    self.priors[name][telescope_idx, filter_idx] = _GaussianParam(mean, stdev)
                else:
                    self.priors[name][telescope_idx, filter_idx, epoch_idx] = _GaussianParam(mean, stdev)

        # Store some info for later
        self.fitting_params.append(name)
        self._filter_idx.append(filter_idx)
        self._epoch_idx.append(epoch_idx)
        self._telescope_idx.append(telescope_idx)

    def _convert_unit_cube(self, cube):
        '''
        Function to convert the unit cube for dynesty into a set of physical
        values

        Notes
        -----
        All conversion from unit q values to actual LD values is done here
        using convert_qtoA. Does not need to be reproduced elsewhere
        '''
        new_cube = np.zeros(len(self.fitting_params))

        skip_params = 0

        for i, name in enumerate(self.fitting_params):
            if skip_params > 0:
                # Skip parameters due to LDC calculations
                skip_params -= 1

            else:
                tidx = self._telescope_idx[i]
                fidx = self._filter_idx[i]
                eidx = self._epoch_idx[i]

                if name in self.limb_dark_coeffs:
                    # Here we handle the different types of limb darkening, and
                    # enforce a physically allowed set of limb darkening coefficients
                    # following Kipping 2013 https://arxiv.org/abs/1308.0009 for
                    # two-parameter limb darkening methods

                    # A quick check to make sure that we are at the start of a
                    # particular filter
                    if not name[1] == '0':
                        raise Exception('There\'s an error in the LCD conversions')

                    # Get all [0,1] values of LDCs for this filter.
                    LDCs = cube[i: i + len(self.limb_dark_coeffs)]

                    # These are all the Kipping parameter, q!
                    # Convert the Kipping parameter q into the physical u
                    #new_cube[i:i + len(self.limb_dark_coeffs)] = self.ld_handler.convert_qtou(*LDCs)
                    new_cube[i:i + len(self.limb_dark_coeffs)] = LDCs

                    # Skip the rest of the LDCs for the filter
                    skip_params = len(self.limb_dark_coeffs) - 1

                else:
                    # Go through all the possibilities of indices
                    if tidx is None:
                        if fidx is None:
                            if eidx is None:
                                new_cube[i] = self.priors[name].from_unit_interval(cube[i])
                            else:
                                new_cube[i] = self.priors[name][eidx].from_unit_interval(cube[i])
                        else:
                            if eidx is None:
                                new_cube[i] = self.priors[name][fidx].from_unit_interval(cube[i])
                            else:
                                new_cube[i] = self.priors[name][fidx, eidx].from_unit_interval(cube[i])
                    else:
                        if fidx is None:
                            if eidx is None:
                                new_cube[i] = self.priors[name][tidx].from_unit_interval(cube[i])
                            else:
                                new_cube[i] = self.priors[name][tidx, eidx].from_unit_interval(cube[i])
                        else:
                            if eidx is None:
                                new_cube[i] = self.priors[name][tidx, fidx].from_unit_interval(cube[i])
                            else:
                                new_cube[i] = self.priors[name][tidx, fidx, eidx].from_unit_interval(cube[i])

        return new_cube

    def _interpret_param_array(self, array):
        '''
        Interprets the parameter cube generated by dynesty and returns
        the parameters in a format usable by the LikelihoodCalculator
        '''
        if not len(array) == len(self.fitting_params):
            raise ValueError('Param array is the wrong length {} for the number of parameters being fitted {}!'.format(len(array), len(self.fitting_params)))

        # First initialise a dictionary to store the physical parameters
        result = {}

        # Now make some empty arrays we can fill in.
        result['rp'] = np.zeros(self.n_filters)

        result['norm'] = np.ones((self.n_telescopes, self.n_filters, self.n_epochs))

        for u in self.limb_dark_coeffs:
            result[u] = np.full(self.n_filters, None)

        for d in self.detrending_coeffs:
            if not isinstance(self.priors[d], _Param):
                # This is not being fitted globally - set up an array
                # We know this as if this is not a _Param instance, it must be
                # an array of _Params
                result[d] = np.full(self.priors[d].shape, None)

        if self.fit_ttv:
            # We need individual t0 for each light curve
            result['t0'] = np.full((self.n_telescopes, self.n_filters, self.n_epochs), None)

        # Now we generate the results

        # First we deal with the parameters we are actually fitting.
        for i, key in enumerate(self.fitting_params):
            tidx = self._telescope_idx[i]
            fidx = self._filter_idx[i]
            eidx = self._epoch_idx[i]

            param_value = array[i]

            # Go through all the possibilities of indices
            if tidx is None:
                if fidx is None:
                    if eidx is None:
                        result[key] = param_value
                    else:
                        result[key][eidx] = param_value
                else:
                    if eidx is None:
                        result[key][fidx] = param_value
                    else:
                        result[key][fidx, eidx] = param_value
            else:
                if fidx is None:
                    if eidx is None:
                        result[key][tidx] = param_value
                    else:
                        result[key][tidx, eidx] = param_value
                else:
                    if eidx is None:
                        result[key][tidx, fidx] = param_value
                    else:
                        result[key][tidx, fidx, eidx] = param_value

        # TODO: something aboout the detrending coeffs - these need to fill up
        # a full (n_telescopes, n_filters, n_epoch) array, even if they're
        # being fitted globally.
        for di, d in enumerate(self.detrending_coeffs):
            # Check the fit mode
            fit_mode = self.detrending_coeffs_fit_mode[di]

            if not fit_mode == 3:
                # This is not fitted by lightcurve - Need to make full array and
                # populate it.
                new_array = np.full((self.n_telescopes, self.n_filters, self.n_epochs), None)

                if fit_mode == 0:
                    # This is a globally fitted parameter
                    new_array = np.full((self.n_telescopes, self.n_filters, self.n_epochs), result[d])
                elif fit_mode == 1:
                    # this is fitted by filter - current shape is (n_filters)
                    for fi in range(self.n_filters):
                        new_array[:,fi,:] = result[d][fi]

                elif fit_mode == 2:
                    # this is fitted by epoch
                    for ei in range(self.n_epochs):
                        new_array[:,:,ei] = result[d][ei]

                else:
                    raise ValueError('Unrecognised detrending coeff fit mode {}'.format(fit_mode))

                result[d] = new_array

        # Now need to check if the LD params are being fitted individually or
        # if they are coupled to the first one.
        if self.fit_ld:
            if self.ld_fit_method == 'single':
                # We need to estimate the values of the remaining LD params
                ld0_values = [result[u][0] for u in self.limb_dark_coeffs]

                est_ld_values = self.ld_handler.ldtk_estimate(ld0_values)

                for i, u in enumerate(self.limb_dark_coeffs):
                    result[u] = est_ld_values[:, i]

        # Now we consider any parameters which aren't being fitted
        # Note that we don't need to consider detrending or normalisation
        # coeffs as they will ONLY be present at all if we are detrending or
        # normalising
        for key in self.priors:
            if key not in result:
                result[key] = self.priors[key].default_value

            elif key in ['rp'] + self.limb_dark_coeffs:
                # We need this bit because we initialised the arrays above
                if len(result[key]) == 0:
                    result[key] = self.priors[key].default_value

        return result

    def fit_detrending(self, lightcurves, method_list, method_index_array,
                       lower_lim=-10000, upper_lim=10000):
        '''
        Initialises detrending for the light curves.

        Parameters
        ----------
        lightcurves : np.array of `LightCurve`s, shape (n_telescopes, n_filters, n_epochs)
            The array of LightCurves to be detrended.
        method_list : array_like, shape (n_detrending_models, 2)
            A list of different detrending models. Each entry should consist
            of a method and a second parameter dependent on the method.
            Accepted methods are
                ['nth order', order]
                ['custom', function, [global fit indices, filter fit indices, epoch fit indices]]
                ['off', ]
            function here is a custom detrending function. TransitFit assumes
            that the first argument to this function is times and that all
            other arguments are single-valued - TransitFit cannot fit
            list/array variables. If 'off' is used, no detrending will be
            applied to the `LightCurve`s using this model.

            If a custom function is used, and some inputs to the function
            should not be fitted individually for each light curve, but should
            instead be shared either globally, within a given filter, or within
            a given epoch, the indices of where these fall within the arguments
            of the detrending function should be given as a list. If there are
            no indices to be given, then use an empty list: []
            e.g. if the detrending function is given by
                ```
                foo(times, a, b, c):
                    # do something
                ```
            and a should be fitted globally, then the entry in the method_list
            would be ['custom', foo, [1], [], []].


        method_index_array : array_like, shape (n_telescopes, n_filters, n_epochs)
            For each LightCurve in `lightcurves`, this array should contain the
            index of the detrending method in `method_list` to be applied to
            this LightCurve
        lower_lim : float, optional
            The lower limit to place on the detrending coefficients.
            Default is -15
        upper_lim : float, optional
            The upper limit to place on the detrending coefficients.
            Default is 15
        '''
        if self.detrend:
            raise ValueError('Detrending is already initialised. You need to make a new PriorInfo to use another detrending method!')

        self._validate_lightcurves_array(lightcurves)

        # Store some info - used in splitting
        self._detrend_method_list = method_list
        self._detrend_method_index_array = method_index_array

        for i in np.ndindex(lightcurves.shape):
            telescope_idx = i[0]
            filter_idx = i[1]
            epoch_idx = i[2]

            if lightcurves[i] is not None:
                # We have a lightcurve - let's detrend it.
                method_idx = method_index_array[i]
                method = method_list[method_idx]
                if not method[0] == 'off':
                    # Set up the detrending in the LightCurve
                    if method[0] == 'nth order':
                        lightcurves[i].set_detrending(method[0], order=method[1])
                    elif method[0] == 'custom':
                        lightcurves[i].set_detrending(method[0], function=method[1])
                    else:
                        raise ValueError('Unable to recognise method list entry {}'.format(method))

                    # Now we set up the fitting of the detrending params
                    n_coeffs = lightcurves[i].n_detrending_params

                    for coeff_i in range(n_coeffs):
                        coeff_name = 'd{}_{}'.format(coeff_i, method_idx)

                        # NOTE: Deal separately with nth order and custom modes
                        if method[0] == 'nth order':
                            # Initialise an entry in the priors dict if there isn't
                            # one already
                            if coeff_name not in self.priors:
                                # Fitting for each lightcurve
                                self.priors[coeff_name] = np.full((self.n_telescopes, self.n_filters, self.n_epochs), None, object)
                                self.detrending_coeffs_fit_mode.append(3)

                                self.detrending_coeffs.append(coeff_name)

                            # Set up the fitting
                            self.add_uniform_fit_param(coeff_name, lower_lim, upper_lim, telescope_idx, filter_idx, epoch_idx)

                        elif method[0] == 'custom':
                            if coeff_name not in self.priors:
                                if coeff_i + 1 in method[2]:
                                    # Fit globally
                                    self.priors[coeff_name] = None
                                    self.detrending_coeffs_fit_mode.append(0)
                                elif coeff_i + 1 in method[3]:
                                    # Fit across filter
                                    self.priors[coeff_name] = np.full((self.n_filters), None, object)
                                    self.detrending_coeffs_fit_mode.append(1)
                                elif coeff_i + 1 in method[4]:
                                    # Fit across epoch
                                    self.priors[coeff_name] = np.full((self.n_epochs), None, object)
                                    self.detrending_coeffs_fit_mode.append(2)
                                else:
                                    # Fitting for each lightcurve
                                    self.priors[coeff_name] = np.full((self.n_telescopes, self.n_filters, self.n_epochs), None, object)
                                    self.detrending_coeffs_fit_mode.append(3)

                                self.detrending_coeffs.append(coeff_name)

                            # Now we set up the fitting
                            if coeff_i + 1 in method[2]:
                                # Fit globally
                                self.add_uniform_fit_param(coeff_name, lower_lim, upper_lim)
                            elif coeff_i + 1 in method[3]:
                                # Fit across filter
                                self.add_uniform_fit_param(coeff_name, lower_lim, upper_lim, filter_idx=filter_idx)
                            elif coeff_i + 1 in method[4]:
                                # Fit across epoch
                                self.add_uniform_fit_param(coeff_name, lower_lim, upper_lim, epoch_idx=epoch_idx)
                            else:
                                # Fitting for each lightcurve
                                self.add_uniform_fit_param(coeff_name, lower_lim, upper_lim, telescope_idx, filter_idx, epoch_idx)

        self.detrend=True

    def fit_limb_darkening(self, fit_method='independent',
                           host_T=None, host_logg=None,
                           host_z=None, filters=None,
                           n_samples=20000, do_mc=False, cache_path=None):
        '''
        Initialises fitting of limb darkening parameters, either independently
        or coupled across wavebands.

        Parameters
        ----------
        fit_method : str, optional
            Determines the mode of fitting of LD parameters. The available
            modes are:
                - 'coupled' : all limb darkening parameters are fitted
                  independently, but are coupled to a wavelength dependent
                  model based on the host parameters
                - 'single' : LD parameters are still tied to a model, but only
                  the first filter is actively fitted. The remaining filters
                  are estimated based off the ratios given by ldtk for a host
                  with the given parameters. This mode is useful for a large
                  number of filters, as 'coupled' or 'independent' fitting will
                  lead to much higher computation times.
                - 'independent' : Each LD coefficient is fitted separately for
                  each filter, with no coupling to the ldtk models.
            The default is 'independent'.
        host_T : tuple or None, optional
            The effective temperature of the host star, in Kelvin, given as a
            (value, uncertainty) pair. Must be provided if fit_method is
            'coupled' or 'single'. Default is None.
        host_logg : tuple or None, optionalor None, optional
            The log_10 of the surface gravity of the host star, with gravity
            measured in cm/s2. Should be given as a (value, uncertainty) pair.
            Must be provided if fit_method is 'coupled' or 'single'. Default is
            None.
        host_z : tuple or None, optional
            The metalicity of the host, given as a (value, uncertainty) pair.
            Must be provided if fit_method is 'coupled' or 'single'. Default is
            None.
        filters : array_like or None, optional
            The set of filters, given in [low, high] limits for the wavelengths
            with the wavelengths given in nanometers. The ordering of the
            filters should correspond to the filter_idx parameter used
            elsewhere. Must be provided if fit_method is 'coupled' or 'single'.
            Default is None.
        n_samples : int, optional
            The number of limb darkening profiles to create. Passed to
            ldtk.LDPSetCreator.create_profiles(). Default is 20000.
        do_mc : bool, optional
            If True, will use MCMC to estimate coefficient uncertainties more
            accurately. Default is False.
        cache_path : str, optional
            This is the path to cache LDTK files to. If not specified, will
            default to the LDTK default
        '''

        # Sanity checks
        if fit_method not in ['coupled', 'single', 'independent']:
            raise ValueError('Unrecognised fit method {}'.format(fit_method))

        if not fit_method == 'independent':
            if filters is None:
                raise ValueError('Filters must be provided for coupled and single ld_fit_methods')
            if not len(filters) == self.n_filters:
                raise ValueError('{} filters were given, but there are {} filters required!'.format(len(filters), self.n_filters))

        # Some flags and useful info to store
        self.fit_ld = True
        self.ld_fit_method = fit_method
        self.filters = filters
        self._n_ld_samples = n_samples
        self._do_ld_mc = do_mc
        self._ld_cache_path = cache_path

        # First up, we need to initialise each LDC for fitting:
        # Note that whist we are fitting
        for i in range(self.n_filters):
            for ldc, name in enumerate(self.limb_dark_coeffs):
                self.add_uniform_fit_param(name, 0, 1, filter_idx=i)

            # If we are in 'single' mode, we only need to fit for the first
            # wavelength
            if self.ld_fit_method == 'single':
                break

        if not fit_method == 'independent':
            # Now if we are coupling across wavelength we must initialise PyLDTK
            self.ld_handler.initialise_ldtk(host_T, host_logg, host_z, filters,
                                            self.limb_dark,
                                            n_samples, do_mc, cache_path)

    def fit_normalisation(self, lightcurves):
        '''
        When run, the Retriever will fit normalisation of the data as a
        free parameter.

        Parameters
        ----------
        lightcurves : np.array of `LightCurve`s, shape (n_telescopes, n_filters, n_epochs)
            The array of LightCurves to be normalised. We can use the value of
            fluxes to estimate a normalisation constant (c_n) range for each
            light curve. We use
                ``1/f_median -1 <= c_n <= 1/f_median + 1``
            as the default range, where f_median is the median flux value.
        default_low : float, optional
            The lowest value to consider as a multiplicative normalisation
            constant. Default is 0.1.
        '''
        if self.normalise:
            raise ValueError('Detrending is already initialised. You need to make a new PriorInfo to use another detrending method!')

        self._validate_lightcurves_array(lightcurves)

        for i in np.ndindex(lightcurves.shape):
            telescope_idx = i[0]
            filter_idx = i[1]
            epoch_idx = i[2]
            if lightcurves[i] is not None:
                # A light curve exists. Set up normalisation
                best, low, high = lightcurves[i].set_normalisation()

                self.add_uniform_fit_param('norm', low, high, telescope_idx, filter_idx, epoch_idx)

        self.normalise = True

    def _validate_lightcurves_array(self, lightcurves):
        '''
        Checks that an array of LightCurves is the correct shape for use with
        the PriorInfo
        '''
        if not lightcurves.shape == (self.n_telescopes, self.n_filters, self.n_epochs):
            raise ValueError('lightcurves has shape {} but should have shape {}'.format(lightcurves.shape, (self.n_telescopes, self.n_filters, self.n_epochs)))

    def __str__(self):
        '''
        Sets the print behaviour for a PriorInfo
        '''
        print_str = 'Priors information:'

        # Add info on memory location of object
        print_str += self.__repr__() + '\n'

        print_str += 'Limb darkening model: {}\n'.format(self.limb_dark)
        print_str += 'n telescopes: {}\n'.format(self.n_telescopes)
        print_str += 'n filters: {}\n'.format(self.n_filters)
        print_str += 'n epochs: {}\n'.format(self.n_epochs)


        # Add the priors, with fitting mode and relevant numbers
        for var in self.priors:
            #Check if iterable
            if isinstance(self.priors[var], Iterable):
                temp_var = np.array(self.priors[var])
                for i in np.ndindex(temp_var.shape):
                    if temp_var[i] is not None:
                        if type(temp_var[i]) is _Param:
                            print_str += '{}, {}: Fixed - value: {}\n'.format(var, i, temp_var[i].default_value)
                        elif type(temp_var[i]) is _UniformParam:
                            print_str += '{}, {}: Uniform - min: {} - max: {}\n'.format(var, i, temp_var[i].low_lim, temp_var[i].high_lim)
                        elif type(temp_var[i]) is _GaussianParam:
                            print_str += '{}, {}: Gaussian - mean: {} - stdev: {}\n'.format(var, i, temp_var[i].mean, temp_var[i].stdev)
                        else:
                            print_str += '{}, {}: Unrecognised type - {}\n'.format(var, i, temp_var[i].__str__())
            else:
                if type(self.priors[var]) is _Param:
                    print_str += '{}: Fixed - value: {}\n'.format(var, self.priors[var].default_value)
                elif type(self.priors[var]) is _UniformParam:
                    print_str += '{}: Uniform - min: {} - max: {}\n'.format(var, self.priors[var].low_lim, self.priors[var].high_lim)
                elif type(self.priors[var]) is _GaussianParam:
                    print_str += '{}: Gaussian - mean: {} - stdev: {}\n'.format(var, self.priors[var].mean, self.priors[var].stdev)
                else:
                    print_str += '{}: Unrecognised type - {}\n'.format(var, self.priors[var].__str__())

        return print_str
