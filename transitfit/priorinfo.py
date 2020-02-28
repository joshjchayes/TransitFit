'''
PriorInfo objects

Object to handle and deal with prior info for retrieval
'''

import numpy as np
from ._params import _Param, _UniformParam
from .detrending_funcs import NthOrderDetrendingFunction
from .detrender import DetrendingFunction
from ._limb_darkening_handler import LimbDarkeningHandler

_prior_info_defaults = {'P':1, 'a':10, 'inc':90, 'rp':0.05, 't0':0, 'ecc':0,
                        'w':90, 'limb_dark':'quadratic', 'q0':0.1, 'q1':0.3,
                        'q2':-0.1, 'q3':0, 'n_telescopes':1,  'n_filters':1,
                        'n_epochs':1, 'norm':1}


def setup_priors(P, rp, a, inc, t0, ecc, w, ld_model, n_telescopes, n_filters, n_epochs, norm=1):
    '''
    Initialises a PriorInfo object with default parameter values. Fitting
    parameters can be added by
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
                    'norm' : norm,
                    'limb_dark' : ld_model}

    return PriorInfo(default_dict, warn=False)

class PriorInfo:
    def __init__(self, default_dict, warn=True):
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

        # Initialise limb darkening
        self.ld_handler = LimbDarkeningHandler(self.limb_dark)
        self.limb_dark_coeffs = self.ld_handler.get_required_coefficients()
        self.fit_ld = False

        for ldc in self.limb_dark_coeffs:
            # Set up the default values for the limb darkening coeffs
            self.priors[ldc] = [_Param(_prior_info_defaults[ldc]) for i in range(self.n_filters)]

        # Initialse a bit for the detrending
        self.detrend = False
        self.detrending_coeffs = []

        # Initialise normalisation things
        self.normalise=False
        self.priors['norm'] = np.ones((self.n_telescopes, self.n_filters, self.n_epochs), object)

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
                self.priors[key] = _Param(default_dict[key])


    def add_uniform_fit_param(self, name, best, low_lim, high_lim,
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

        # Deal with wavelength variant parameters
        if name in ['rp'] + self.limb_dark_coeffs:
            if filter_idx is None:
                raise ValueError('filter_idx must be provided for parameter {}'.format(name))
            self.priors[name][filter_idx] = _UniformParam(best, low_lim, high_lim)

        # Deal with detrending fitting
        elif name in self.detrending_coeffs + ['norm']:
            if filter_idx is None:
                raise ValueError('filter_idx must be provided for parameter {}'.format(name))
            if epoch_idx is None:
                raise ValueError('epoch_idx must be provided for parameter {}'.format(name))

            self.priors[name][telescope_idx, filter_idx, epoch_idx] = _UniformParam(best, low_lim, high_lim)

        # Anything else
        # Note t0 is included in here - we just need t0 from one light curve
        # to be able to fit that with P
        else:
            self.priors[name] = _UniformParam(best, low_lim, high_lim)

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

                if name in ['rp']:
                    new_cube[i] = self.priors[name][fidx].from_unit_interval(cube[i])

                elif name in self.detrending_coeffs + ['norm']:
                    new_cube[i] = self.priors[name][tidx, fidx, eidx].from_unit_interval(cube[i])

                elif name in self.limb_dark_coeffs:
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

                    new_cube[i:i + len(self.limb_dark_coeffs)] = self.ld_handler.convert_qtoA(*LDCs)

                    # Skip the rest of the LDCs for the filter
                    skip_params = len(self.limb_dark_coeffs) - 1

                else:
                    new_cube[i] = self.priors[name].from_unit_interval(cube[i])

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
            result[d] = np.full((self.n_telescopes, self.n_filters, self.n_epochs), None)

        # Now we generate the results

        # First we deal with the parameters we are actually fitting.
        for i, key in enumerate(self.fitting_params):
            tidx = self._telescope_idx[i]
            fidx = self._filter_idx[i]
            eidx = self._epoch_idx[i]

            if key in ['rp'] + self.limb_dark_coeffs:
                result[key][fidx] = array[i]

            elif key in self.detrending_coeffs + ['norm']:
                result[key][tidx, fidx, eidx] = array[i]

            else:
                result[key] = array[i]

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

    def _interpret_results_array(self, array):
        '''
        Interprets the final results (best params) so that we can quickly plot
        '''
        pass

    def add_detrending(self, lightcurves, method_list, method_index_array,
                       lower_lim=-15, upper_lim=15, function=None,
                       coeff_lims=None):
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
                ['custom', function]
                ['off', ]
            function here is a custom detrending function. TransitFit assumes
            that the first argument to this function is times and that all
            other arguments are single-valued - TransitFit cannot fit
            list/array variables. If 'off' is used, no detrending will be
            applied to the `LightCurve`s using this model.
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
                    n_coeffs = lightcurves[i].num_detrending_params

                    for coeff_i in range(n_coeffs):
                        coeff_name = 'd{}_{}'.format(coeff_i, method_idx)

                        # Initialise an entry in the priors dict if there isn't
                        # one already
                        if coeff_name not in self.priors:
                            self.priors[coeff_name] = np.full((self.n_telescopes, self.n_filters, self.n_epochs), None, object)

                            self.detrending_coeffs.append(coeff_name)

                        self.add_uniform_fit_param(coeff_name, (lower_lim + upper_lim)/2, lower_lim, upper_lim, telescope_idx, filter_idx, epoch_idx)

        self.detrend=True

    def fit_limb_darkening(self, fit_method='independent', low_lim=-5,
                           high_lim=-5, host_T=None, host_logg=None,
                           host_z=None, filters=None, ld_model='quadratic',
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
        low_lim : float, optional
            The lower limit to use in conversion in the case where there are
            open bounds on a coefficient (power2 and nonlinear models). Note
            that in order to conserve sampling density in all regions for the
            power2 model, you should set lower_lim=-high_lim. Default is -5
        high_lim : float, optional
            The upper limit to use in conversion in the case where there are
            open bounds on a coefficient (power2 and nonlinear models). Note
            that in order to conserve sampling density in all regions for the
            power2 model, you should set lower_lim=-high_lim. Default is 5
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
        ld_model : str, optional
            The model of limb darkening to use. Default is 'quadratic'
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

        # First up, we need to initialise each LDC for fitting:
        for i in range(self.n_filters):
            for ldc, name in enumerate(self.limb_dark_coeffs):
                self.add_uniform_fit_param(name, 0.5, 0, 1, filter_idx=i)

            # If we are in 'single' mode, we only need to fit for the first
            # wavelength
            if self.ld_fit_method == 'single':
                break

        if not fit_method == 'independent':
            # Now if we are coupling across wavelength we must initialise PyLDTK
            self.ld_handler.initialise_ldtk(host_T, host_logg, host_z, filters,
                                            ld_model, n_samples, do_mc,
                                            cache_path)

    def fit_normalisation(self, lightcurves, default_low=0.1):
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
                best, low, high = lightcurves[i].set_normalisation(default_low)

                self.add_uniform_fit_param('norm', best, low, high, telescope_idx, filter_idx, epoch_idx)

        self.normalise = True

    def _validate_lightcurves_array(self, lightcurves):
        '''
        Checks that an array of LightCurves is the correct shape for use with
        the PriorInfo
        '''
        if not lightcurves.shape == (self.n_telescopes, self.n_filters, self.n_epochs):
            raise ValueError('lightcurves has shape {} but should have shape {}'.format(lightcurves.shape, (self.n_telescopes, self.n_filters, self.n_epochs)))
