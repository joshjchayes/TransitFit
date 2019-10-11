'''
PriorInfo objects

Object to handle and deal with prior info for retrieval
'''

import numpy as np
from ._params import _Param, _UniformParam
from .detrending_funcs import linear, quadratic, sinusoidal
from .detrender import DetrendingFunction
from ._ld_param_handler import LDParamHandler

_prior_info_defaults = {'P':1, 'a':10, 'inc':90, 'rp':0.05, 't0':0, 'ecc':0,
                        'w':90, 'limb_dark':'quadratic', 'u1':0.1, 'u2':0.3,
                        'u3':-0.1, 'u4':0, 'num_wavelengths':1,
                        'num_times':1, 'norm' : 1}

_detrending_functions = {'linear' : linear, 'quadratic' : quadratic,
                         'sinusoidal' : sinusoidal}

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

        self.fitting_params = []
        self._filter_idx = []
        self._epoch_idx = []
        self.priors = {}
        self.num_times = default_dict['num_times']
        self.num_wavelengths = default_dict['num_wavelengths']
        self.limb_dark = default_dict['limb_dark']

        # Sort out the limb darkening coefficients - what is allowed?
        if self.limb_dark == 'uniform':
            self.limb_dark_coeffs = []
        if self.limb_dark == 'linear':
            self.limb_dark_coeffs = ['u1']
        elif self.limb_dark in ['quadratic', 'logarithmic', 'exponential','squareroot', 'power2']:
            self.limb_dark_coeffs = ['u1', 'u2']
        elif self.limb_dark == 'nonlinear':
            self.limb_dark_coeffs = ['u1', 'u2', 'u3', 'u4']
        else:
            raise ValueError('Unrecognised limb darkening method {}'.format(self.limb_dark))

        allowed_coeff_names = ['u1','u2','u3','u4']
        for key in allowed_coeff_names:
            if key not in self.limb_dark_coeffs:
                default_dict.pop(key, None)

        self.fit_ld = False

        # Set up a dictionary containing the parameters
        for key in default_dict.keys():
            if key in ['rp'] + self.limb_dark_coeffs:
                self.priors[key] = [_Param(default_dict[key]) for i in range(self.num_wavelengths)]
            elif key =='t0':
                self.priors[key] = [_Param(default_dict[key]) for i in range(self.num_times)]
            elif key in ['num_times', 'num_wavelengths', 'limb_dark']:
                pass
            elif key in ['norm']:
                self.priors[key] = np.array([[_Param(default_dict[key]) for i in range(self.num_times)] for j in range(self.num_wavelengths)], object)
            else:
                self.priors[key] = _Param(default_dict[key])

        # Initialse a bit for the detrending
        self.detrend = False
        self.detrending_coeffs = []

        # Initialise normalisation things
        self.normalise=False
        self.priors['norm'] = np.ones((self.num_wavelengths, self.num_times))
        self.priors['shift'] = np.ones((self.num_wavelengths, self.num_times))

    def add_uniform_fit_param(self, name, best, low_lim, high_lim, epoch_idx=None, filter_idx=None):
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

        # Deal with epoch variant parameters
        elif name in ['t0']:
            if epoch_idx is None:
                raise ValueError('epoch_idx must be provided for parameter {}'.format(name))
            self.priors[name][epoch_idx] = _UniformParam(best, low_lim, high_lim)

        # Deal with detrending fitting
        elif name in self.detrending_coeffs:
            if filter_idx is None:
                raise ValueError('filter_idx must be provided for parameter {}'.format(name))
            if epoch_idx is None:
                raise ValueError('epoch_idx must be provided for parameter {}'.format(name))

            self.priors[name][filter_idx, epoch_idx] = _UniformParam(best, low_lim, high_lim)

        # Anything else
        else:
            self.priors[name] = _UniformParam(best, low_lim, high_lim)

        # Store some info for later
        self.fitting_params.append(name)
        self._filter_idx.append(filter_idx)
        self._epoch_idx.append(epoch_idx)

    def _from_unit_interval(self, i, u):
        '''
        Gets parameter self.fitting_params[i] from a number u between 0 and 1
        '''
        name = self.fitting_params[i]
        fidx = self._filter_idx[i]
        eidx = self._epoch_idx[i]

        if name in ['rp'] + self.limb_dark_coeffs:
            return self.priors[name][fidx].from_unit_interval(u)

        elif name in ['t0']:
            return self.priors[name][eidx].from_unit_interval(u)

        elif name in self.detrending_coeffs:
            return self.priors[name][fidx, eidx].from_unit_interval(u)

        return self.priors[name].from_unit_interval(u)

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
        result['rp'] = np.zeros(self.num_wavelengths)
        result['t0'] = np.zeros(self.num_times)

        result['norm'] = np.ones((self.num_wavelengths, self.num_times))
        result['shift'] = np.ones((self.num_wavelengths, self.num_times))

        for u in self.limb_dark_coeffs:
            result[u] = np.zeros(self.num_wavelengths)

        for d in self.detrending_coeffs:
            result[d] = np.zeros((self.num_wavelengths, self.num_times))

        # Now we generate the results

        # First we deal with the parameters we are actually fitting.
        for i, key in enumerate(self.fitting_params):
            fidx = self._filter_idx[i]
            eidx = self._epoch_idx[i]

            if key in ['rp'] + self.limb_dark_coeffs:
                result[key][fidx] = array[i]

            elif key in ['t0']:
                result[key][eidx] = array[i]

            elif key in self.detrending_coeffs + ['norm', 'shift']:
                result[key][fidx, eidx] = array[i]

            else:
                result[key] = array[i]

        # Now need to check if the LD params are being fitted individually or
        # if they are coupled to the first one.
        if self.fit_ld:
            if self.ld_fit_method == 'single':
                # We need to estimate the values of the remaining LD params
                ld0_values = [result[u][0] for u in self.limb_dark_coeffs]

                est_ld_values = self.ld_param_handler.estimate_values(ld0_values,
                self.ld_param_handler.default_model)

                for i, u in enumerate(self.limb_dark_coeffs):
                    result[u] = est_ld_values[:, i]



        # Now we consider any parameters which aren't being fitted
        # Note that we don't need to consider detrending  or normalisation
        # coeffs as they will ONLY be present at all if we are detrending or
        # normalising
        for key in self.priors:
            if key not in result:
                result[key] = self.priors[key].default_value

            elif key in ['rp','t0'] + self.limb_dark_coeffs:
                # We need this bit because we initialised the arrays above
                if len(result[key]) == 0:
                    result[key] = self.priors[key].default_value

        return result

    def _interpret_results_array(self, array):
        '''
        Interprets the final results (best params) so that we can quickly plot
        '''
        pass

    def add_detrending(self, data_array, method='linear'):
        '''
        Initialises detrending for the light curves.

        Parameters
        ----------
        data_array : np.array
            One of the times, flux or uncertainty arrays to be used in fitting.
            This is required to ensure that we only detrend light curves which
            actually exist!
        method : str, optional
            The detrending method to use. Accepted are linear, quadratic and
            sinusoidal.
        '''
        method = method.lower()
        if method not in ['linear','quadratic','sinusoidal']:
            raise ValueError('Unrecognised detrending method {}. Acceptable are linear, quadratic, sinusoidal'.format(method))

        if self.detrend:
            raise ValueError('Detrending is already initialised. You need to make a new PriorInfo to use another detrending method!')

        self.detrend = True

        # Set up the detrending function
        self.detrending_method = method
        self.detrending_function = DetrendingFunction(_detrending_functions[method])

        # Set up the parameters to be detrended
        # The -1 is there because we assume that the first arg is the times.
        for di in range(self.detrending_function.n_required_args - 1):
            key = 'd{}'.format(di)  # What is the parameter referred to as?

            # Initialse a default priors array.
            self.priors[key] = np.array([[None for i in range(self.num_times)] for j in range(self.num_wavelengths)], object)

            # Add to an array of listing valid detrending coeffs
            self.detrending_coeffs.append(key)

            # Add the fitting info!
            for j in range(self.num_wavelengths):
                for k in range(self.num_times):
                    if data_array[j, k] is not None:
                        # A light curve exists
                        if not (method == 'sinusoidal' and di == 2):
                            self.add_uniform_fit_param(key, 0, -5, 5, filter_idx=j, epoch_idx=k)
                        else:
                             # Limit 0 < d2 <= 2pi for sinusoidal
                             self.add_uniform_fit_param(key, 0, 0, 2*np.pi, filter_idx=j, epoch_idx=k)

    def fit_limb_darkening(self, fit_method='independent', host_T=None, host_logg=None,
                           host_z=None, filters=None, ld_model='quadratic',
                           n_samples=20000, do_mc=False, allowed_variance=5):
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
        ld_model : str, optional
            The model of limb darkening to use. Default is 'quadratic'
        n_samples : int, optional
            The number of limb darkening profiles to create. Passed to
            ldtk.LDPSetCreator.create_profiles(). Default is 20000.
        do_mc : bool, optional
            If True, will use MCMC to estimate coefficient uncertainties more
            accurately. Default is False.
        allowed_variance : float, optional
            The number of standard deviations that each parameter is allowed to
            vary by. Default is 5
        '''

        # Sanity checks
        if fit_method not in ['coupled', 'single', 'independent']:
            raise ValueError('Unrecognised fit method {}'.format(fit_method))

        if not len(filters) == self.num_wavelengths:
            raise ValueError('{} filters were given, but there are {} filters required!'.format(len(filters), self.num_wavelengths))

        # Some flags and useful info to store
        self.fit_ld = True
        self.ld_fit_method = fit_method

        if fit_method == 'independent':
            # We fit each LD parameter independently without any coupling
            for i in range(self.num_wavelengths):
                for ldi, name in enumerate(self.limb_dark_coeffs):
                    self.add_uniform_fit_param(name, 0, -1, 1, filter_idx=i)

        else:
            # We need to couple LD params across wavelengths

            # Check required parameters are given.
            #if


            # Create the LDParamHandler
            self.ld_param_handler = LDParamHandler(host_T, host_logg, host_z,
                                                   filters, ld_model, n_samples,
                                                   do_mc)

            # Work out how many parameters we need to fit for and initialise the
            # fitting for them.
            for i in range(self.num_wavelengths):
                for ldi, name in enumerate(self.limb_dark_coeffs):
                    best = self.ld_param_handler.coeffs[ld_model][0][i][ldi]
                    err = self.ld_param_handler.coeffs[ld_model][1][i][ldi]
                    low = best - allowed_variance * err
                    high = best + allowed_variance * err
                    # allow each parameter to vary by up to 5 sigma

                    self.add_uniform_fit_param(name, best, low, high, filter_idx=i)
                if self.ld_fit_method == 'single':
                    # Stop if we are only fitting the first one
                    break

    def fit_normalisation(self, flux_array, default_low=0.1):
        '''
        When run, the Retriever will fit normalisation of the data as a
        free parameter.

        Parameters
        ----------
        flux_array : np.array
            The array of fluxes to be used in fitting. This serves two purposes
            First, to ensure that we only normalise light curves which
            actually exist! Second, we can use the value of fluxes to
            estimate a normalisation constant (c_n) range for each light curve.
            We use
                ``1/f_median -1 <= c_n <= 1/f_median + 1``
            as the default range, where f_median is the median flux value.
        default_low : float, optional
            The lowest value to consider as a multiplicative normalisation
            constant. Default is 0.1.
        high : float, optional
            The highest value to consider as a multiplicative normalisation
            constant. Default is 15.

        '''
        self.normalise = True

        # like detrending, we have to normalise each light curve separately
        for j in range(self.num_wavelengths):
            for k in range(self.num_times):
                if flux_array[j, k] is not None:
                    # A light curve exists
                    # First we set up the scaling factor
                    med_fact = 1/np.median(flux_array[j, k])
                    if med_fact - 2 <=0:
                        low_fact = default_low
                    else:
                        low_fact = med_fact - 2

                    high_fact = med_fact + 2

                    print('The automatically detected limits for normalisation constants are:')
                    print('Low      Median      High')
                    print(round(low_fact,2), round(med_fact,2),  round(high_fact,2))

                    self.add_uniform_fit_param('norm', med_fact, low_fact, high_fact, filter_idx=j, epoch_idx=k)

                    # Now we fit the shift
                    self.add_uniform_fit_param('shift', 0, -10, 10, filter_idx=j, epoch_idx=k)
