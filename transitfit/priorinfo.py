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
                        'q2':-0.1, 'q3':0, 'num_wavelengths':1,
                        'num_times':1, 'norm' : 1}


def setup_priors(P, rp, a, inc, t0, ecc, w, ld_model, num_times, num_wavelengths, norm=1):
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
                    'num_times' : num_times,
                    'num_wavelengths' : num_wavelengths,
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

        self.fitting_params = []
        self._filter_idx = []
        self._epoch_idx = []
        self.priors = {}
        self.num_times = default_dict['num_times']
        self.num_wavelengths = default_dict['num_wavelengths']
        self.limb_dark = default_dict['limb_dark']

        # Initialise limb darkening
        self.ld_handler = LimbDarkeningHandler(self.limb_dark)
        self.limb_dark_coeffs = self.ld_handler.get_required_coefficients()
        self.fit_ld = False

        for ldc in self.limb_dark_coeffs:
            # Set up the default values for the limb darkening coeffs
            self.priors[ldc] = [_Param(_prior_info_defaults[ldc]) for i in range(self.num_wavelengths)]

        # Initialse a bit for the detrending
        self.detrend = False
        self.detrending_coeffs = []

        # Initialise normalisation things
        self.normalise=False
        self.priors['norm'] = np.ones((self.num_wavelengths, self.num_times), object)

        # Set up a dictionary containing the parameters with default values
        for key in default_dict.keys():
            if key in ['rp'] + self.limb_dark_coeffs:
                self.priors[key] = [_Param(default_dict[key]) for i in range(self.num_wavelengths)]
            #elif key =='t0':
            #    self.priors[key] = [_Param(default_dict[key]) for i in range(self.num_times)]
            elif key in ['num_times', 'num_wavelengths', 'limb_dark']:
                pass
            elif key in ['norm']:
                self.priors[key] = np.array([[_Param(default_dict[key]) for i in range(self.num_times)] for j in range(self.num_wavelengths)], object)
            else:
                self.priors[key] = _Param(default_dict[key])


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

        # Deal with detrending fitting
        elif name in self.detrending_coeffs + ['norm']:
            if filter_idx is None:
                raise ValueError('filter_idx must be provided for parameter {}'.format(name))
            if epoch_idx is None:
                raise ValueError('epoch_idx must be provided for parameter {}'.format(name))

            self.priors[name][filter_idx, epoch_idx] = _UniformParam(best, low_lim, high_lim)

        # Anything else
        # Note t0 is included in here - we just need t0 from one light curve
        # to be able to fit that with P
        else:
            self.priors[name] = _UniformParam(best, low_lim, high_lim)

        # Store some info for later
        self.fitting_params.append(name)
        self._filter_idx.append(filter_idx)
        self._epoch_idx.append(epoch_idx)

    def _from_unit_interval(self, i, u):
        '''
        Gets parameter self.fitting_params[i] from a number u between 0 and 1

        DEPRECIATED as of v0.8
        '''
        name = self.fitting_params[i]
        fidx = self._filter_idx[i]
        eidx = self._epoch_idx[i]

        if name in ['rp']:
            return self.priors[name][fidx].from_unit_interval(u)

        elif name in self.detrending_coeffs + ['norm']:
                return self.priors[name][fidx, eidx].from_unit_interval(u)

        elif name in self.limb_dark_coeffs:
            # Here we handle the different types of limb darkening, and
            # enforce a physically allowed set of limb darkening coefficients
            # following Kipping 2013 https://arxiv.org/abs/1308.0009 for
            # two-parameter limb darkening methods

            # First up, convert the unit-interval fitted

            if not self.ld_fit_method == 'independent':
                # We are using PyLDTK to deal with likelihoods
                return self.priors[name][fidx].from_unit_interval(u)

        return self.priors[name].from_unit_interval(u)

    def _convert_unit_cube(self, cube):
        '''
        Function to convert the unit cube for dynesty into a set of physical
        values
        '''
        new_cube = np.zeros(len(self.fitting_params))

        skip_params = 0

        for i, name in enumerate(self.fitting_params):
            if skip_params > 0:
                # Skip parameters due to LDC calculations
                skip_params -= 1

            else:
                fidx = self._filter_idx[i]
                eidx = self._epoch_idx[i]

                if name in ['rp']:
                    new_cube[i] = self.priors[name][fidx].from_unit_interval(cube[i])

                elif name in self.detrending_coeffs + ['norm']:
                    new_cube[i] = self.priors[name][fidx, eidx].from_unit_interval(cube[i])

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

                    new_cube[i:i + len(self.limb_dark_coeffs)] = self.ld_handler.convert_coefficients(*LDCs)

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
        result['rp'] = np.zeros(self.num_wavelengths)

        result['norm'] = np.ones((self.num_wavelengths, self.num_times))

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

            elif key in self.detrending_coeffs + ['norm']:
                result[key][fidx, eidx] = array[i]

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

    def _add_nth_order_detrending(self, data_array, order=1, lower_lim=-5,
                                  upper_lim=5):
        '''
        Initialises nth order detrending for the light curves.

        Parameters
        ----------
        data_array : np.array
            One of the times, flux or uncertainty arrays to be used in fitting.
            This is required to ensure that we only detrend light curves which
            actually exist!
        order : int, optional
            The order of detrending function to use. Default is 1
        '''
        if self.detrend:
            raise ValueError('Detrending is already initialised. You need to make a new PriorInfo to use another detrending method!')

        order = int(order)

        self.detrend = True
        self.detrending_method = 'nth order'
        # Set up the detrending function
        self.detrending_order = order
        self.detrending_function = DetrendingFunction(NthOrderDetrendingFunction(order))

        # Set up the parameters to be detrended
        for di in range(order):
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
                        self.add_uniform_fit_param(key, (lower_lim + upper_lim)/2, lower_lim, upper_lim, filter_idx=j, epoch_idx=k)

    def _add_custom_detrending(self, data_array, function, coeff_lims=None):
        '''
        Adds custom detrending. We assume that the first argument in function
        is times

        Parameters
        ----------
        data_array : np.array
            One of the times, flux or uncertainty arrays to be used in fitting.
            This is required to ensure that we only detrend light curves which
            actually exist!
        function : function
            The detrending function. We assume that the first argument is
            the times, and that all others are single valued - TransitFit
            cannot fit list/array variables
        coeff_lims : None or array_like, shape (n_args, 2)
            The lower and upper limits of each of the non-time arguments of
            the detrending function.

        '''
        if self.detrend:
            raise ValueError('Detrending is already initialised. You need to make a new PriorInfo to use another detrending method!')

        self.detrend=True
        self.detrending_method = 'custom'
        self.detrending_function = DetrendingFunction(function)

        # Sort out coefficient limits
        if coeff_lims is None:
            # Add in a default upper and lower value of ±100
            coeff_lims = np.array([[-100, 100] for i in range(self.detrending_function.n_required_args)])
        coeff_lims = np.array(coeff_lims)
        if not coeff_lims.shape[0] == self.detrending_function.n_required_args and coeff_lims.shape[1] == 2:
            raise ValueError('Invalid shape {} for coefficient limits with {} required coefficients'.format(coeff_lims.shape, self.detrending_function.n_required_args))

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
                        self.add_uniform_fit_param(key, (coeff_lims[i, 0] + coeff_lims[i, 1])/2, coeff_lims[i, 0], coeff_lims[i, 1], filter_idx=j, epoch_idx=k)


    def add_detrending(self, data_array, method='nth order', order=1,
                       lower_lim=-5, upper_lim=5, function=None,
                       coeff_lims=None):
        '''
        Initialises detrending for the light curves.

        Parameters
        ----------
        data_array : np.array
            One of the times, flux or uncertainty arrays to be used in fitting.
            This is required to ensure that we only detrend light curves which
            actually exist!
        method : str, optional
            The detrending method to use. Accepted are
                nth order
                custom
        order : int, optional
            If nth order detrending specified, this is the order of the
            detrending function. Default is 1.
        lower_lim : float, optional
            The lower limit to place on the coefficients for nth order
            detrending. Default is -5
        upper_lim : float, optional
            The upper limit to place on the coefficients for nth order
            detrending. Default is 5
        function : None or function, optional
            The detrending function. If provided and method is 'custom', will
            apply this as the detrending function. We assume that the first
            argument is times, and that all others are single valued -
            TransitFit cannot fit list/array variables. Default is None
        coeff_lims : None or array_like, shape (n_args, 2)
            The lower and upper limits of each of the non-time arguments of
            the detrending function, if supplied. If None and method is
            'custom', all lower and upper bounds will default to -100 and 100
            respectively.
        '''
        if method == 'nth order':
            self._add_nth_order_detrending(data_array, order, lower_lim, upper_lim)
        elif method == 'custom':
            if function is None:
                raise ValueError('You need to supply a function for custom detrending!')
            self._add_custom_detrending(data_array, function, coeff_lims)

        else:
            raise ValueError('Unrecognised detrending method {}'.format(method))

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
            if not len(filters) == self.num_wavelengths:
                raise ValueError('{} filters were given, but there are {} filters required!'.format(len(filters), self.num_wavelengths))

        # Some flags and useful info to store
        self.fit_ld = True
        self.ld_fit_method = fit_method

        # First up, we need to initialise each LDC for fitting:
        for i in range(self.num_wavelengths):
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
        '''
        self.normalise = True

        # like detrending, we have to normalise each light curve separately
        for j in range(self.num_wavelengths):
            for k in range(self.num_times):
                if flux_array[j, k] is not None:
                    # A light curve exists
                    # First we set up the scaling factor
                    med_fact = 1/np.median(flux_array[j, k])
                    if med_fact - 1 <=0:
                        low_fact = default_low
                    else:
                        low_fact = med_fact - 1

                    high_fact = med_fact + 1

                    #print('The automatically detected limits for normalisation constants are:')
                    #print('Low      Median      High')
                    #print(round(low_fact,2), round(med_fact,2),  round(high_fact,2))

                    self.add_uniform_fit_param('norm', med_fact, low_fact, high_fact, filter_idx=j, epoch_idx=k)
