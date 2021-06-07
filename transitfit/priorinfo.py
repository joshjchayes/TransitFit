'''
Object to handle and deal with prior info for retrieval
'''

import numpy as np
from collections.abc import Iterable

from ._params import _Param, _UniformParam, _GaussianParam
from .detrending_funcs import NthOrderDetrendingFunction
from .detrender import DetrendingFunction
from ._limb_darkening_handler import LimbDarkeningHandler
from ._paramarray import ParamArray

# Parameters which do not vary with telescope, filter, or epoch
# Note that t0 is in here, as it is global unless ttv mode is on
_global_params = ['P', 'a', 'inc', 'ecc', 'w', 't0']

# Parameters which vary with filter
_filter_dependent_params = ['rp', 'q0', 'q1', 'q2', 'q3']

# Params which vary with epoch
_epoch_dependent_params = []

# The possible ldc lables
_all_ldcs = ['q0', 'q1', 'q2', 'q3']

# Default values for parameters (used in io.parse_priors_list)
_prior_info_defaults = {'P':1, 'a':10, 'inc':90, 'rp':0.05, 't0':0, 'ecc':0,
                        'w':90, 'limb_dark':'quadratic', 'q0':0.1, 'q1':0.3,
                        'q2':None, 'q3':None, 'n_telescopes':1,  'n_filters':1,
                        'n_epochs':1, 'norm':1}

def setup_priors(P, t0, a, rp, inc, ecc, w, limb_dark, n_telescopes, n_filters,
                 n_epochs, q0=None, q1=None, q2=None, q3=None, allow_ttv=False,
                 lightcurves=None):
    '''
    Factory function to initialise a PriorInfo object

    qX should either be a single value or a list of values length n_filters
    '''
    # Deal with the qX values
    if q0 is not None and not isinstance(q0, Iterable):
        q0 = [q0 for i in range(n_filters)]
    if q1 is not None and not isinstance(q1, Iterable):
        q1 = [q1 for i in range(n_filters)]
    if q2 is not None and not isinstance(q2, Iterable):
        q2 = [q2 for i in range(n_filters)]
    if q3 is not None and not isinstance(q3, Iterable):
        q3 = [q3 for i in range(n_filters)]

    default_dict = {'P':P, 't0':t0, 'a':a, 'rp':rp, 'inc':inc, 'ecc':ecc,
                    'w':w, 'q0':q0, 'q1':q1, 'q2':q2, 'q3':q3, 'norm':1}

    return PriorInfo(default_dict, limb_dark, n_telescopes, n_filters,
                      n_epochs, allow_ttv, lightcurves)



class PriorInfo:
    '''
    Object to deal with anything involving priors

    Parameters
    ----------
    default_dict : dictionary
        Dictionary containing default parameter values.
    limb_dark : str
        The limb darkening model to use.
    n_telescopes : int
        The number of different telescopes used in the data set
    n_filters : int
        The number of different filters used in the data set
    n_epochs : int
        The number of different epochs used in the data set
    allow_ttv : bool, optional
        If True, will fit t0 to each transit. Default is False.
    lightcurves : array_like, shape(n_telescopes, n_filters, n_epochs)
        Array of the light curves. If there is no observation for a particular
        telescope, filter, epoch combination, then the value should be set to
        None.
    '''
    def __init__(self, default_dict, limb_dark, n_telescopes, n_filters,
                 n_epochs, allow_ttv=False, lightcurves=None):
        # Store the basics
        self.limb_dark = limb_dark
        self.n_telescopes = n_telescopes
        self.n_filters = n_filters
        self.n_epochs = n_epochs
        self.allow_ttv = allow_ttv

        # Initialise limb darkening (set to off)
        self.ld_handler = LimbDarkeningHandler(self.limb_dark)
        self.limb_dark_coeffs = self.ld_handler.get_required_coefficients()
        self.fit_ld = False
        self.ld_fit_method = 'off'

        # Initialse detrending (set to off)
        self.detrend = False
        self.detrending_coeffs = []

        # Initialse normalisation (set to off)
        self.normalise=False

        #####################
        # Set up the priors #
        #####################
        # Dictionary containing info on all the priors
        self.priors = {}

        for key in default_dict.keys():
            #print('Param:', key)
            # Loop through each parameter and initialise the prior


            if key in _global_params:
                # These are global parameters
                if key == 't0' and allow_ttv:
                    # We are fitting a separate t0 for each epoch
                    self.priors[key] = ParamArray(key, (1,1, self.n_epochs), False, False, True, default_dict[key])
                else:
                    self.priors[key] = ParamArray(key, (1,1, 1), False, False, False, default_dict[key])

            elif key in _filter_dependent_params:
                # These parameters vary with filter
                shape = (1, self.n_filters, 1)

                # Deal with the ldcs
                if key in _all_ldcs:
                    if key in self.limb_dark_coeffs:
                        #print('Initialising ldc', key)
                        self.priors[key] = ParamArray(key, shape, False, True, False, default_dict[key][0])
                        for qi, q in enumerate(default_dict[key]):
                            self.priors[key].set_value(q, filter_idx=qi)

                # Otherwise, this is rp
                else:
                    self.priors[key] = ParamArray(key, shape, False, True, False, default_dict[key])

            else:
                # This is a lightcurve dependent parameter
                self.priors[key] = ParamArray(key, (self.n_telescopes, self.n_filters,self.n_epochs), True, True, True, default_dict[key], lightcurves=lightcurves)

        # All priors should now be initialised

        # We also need a list for keeping track of what is being fitted
        # Each entry in this will be
        # [param name, telescope index, filter index, epoch index]
        self.fitting_params = None

        #for p in self.priors.keys():
        #    print(p, self.priors[p].array)

    ###############################################################
    #                   ADDING FIT PARAMS                         #
    ###############################################################

    def add_uniform_fit_param(self, name, low_lim, high_lim,
                              telescope_idx=None, filter_idx=None, epoch_idx=None):
        '''
        Adds a new parameter which will be fitted uniformly in the range given
        by low_lim and high_lim
        '''
        if name in ['a', 'P', 'rp', 'inc', 'ecc', 'w']:
            negative_allowed = False
        else:
            negative_allowed = True

        self.priors[name].add_uniform_fit_param(low_lim, high_lim,
                                                telescope_idx, filter_idx,
                                                epoch_idx, negative_allowed)

        # Store some info for later
        if self.fitting_params is None:
            self.fitting_params = np.array([[name, telescope_idx, filter_idx, epoch_idx]], dtype=object)
        else:
            self.fitting_params = np.append(self.fitting_params, np.array([[name, telescope_idx, filter_idx, epoch_idx]], object), axis=0)
        #self.fitting_params.append([name, telescope_idx, filter_idx, epoch_idx])

    def add_gaussian_fit_param(self, name, mean, stdev,
                               telescope_idx=None, filter_idx=None, epoch_idx=None):
        '''
        Adds a new parameter which will be fitted with a Gaussian prior
        '''
        # If this is t0 and ttvs are on, we want to make the prior much wider to allow for proper TTVs
        #if name == 't0' and self.allow_ttv:
            # Set the Gaussian width to 0.1 days for ttv fitting mode
            #stdev = 0.1
        if name in ['a', 'P', 'rp', 'inc', 'ecc', 'w']:
            negative_allowed = False
        else:
            negative_allowed = True

        self.priors[name].add_gaussian_fit_param(mean, stdev,
                                                 telescope_idx, filter_idx,
                                                 epoch_idx, negative_allowed)

        # Store some info for later
        if self.fitting_params is None:
            self.fitting_params = np.array([[name, telescope_idx, filter_idx, epoch_idx]], dtype=object)
        else:
            self.fitting_params = np.append(self.fitting_params, np.array([[name, telescope_idx, filter_idx, epoch_idx]], object), axis=0)


    ###############################################################
    #         ADDING DETRENDING/NORMALISATION/LDC FITTING         #
    ###############################################################

    def fit_detrending(self, lightcurves, method_list, method_index_array,
                       limits=None):
        '''
        Intialises detrending
        '''
        if self.detrend:
            raise ValueError('Detrending is already initialised. You need to make a new PriorInfo to use another detrending method!')

        # Store some info - used in splitting
        self._detrend_method_list = method_list
        self._detrend_method_index_array = method_index_array

        self.detrending_coeffs = [[] for method in method_list]

        # Go through each lightcurve and initialise the relevant detrending
        for i in np.ndindex(lightcurves.shape):
            telescope_idx, filter_idx, epoch_idx = i

            if lightcurves[i] is not None:
                # We have a light curve, pull out the method info
                method_idx = method_index_array[i]
                method = method_list[method_idx]

                if method[0] == 'off':
                    # No detrending - skip this curve
                    pass
                else:
                    # First we set up the detrending for the lightcurve
                    if method[0] == 'nth order':
                        lightcurves[i].set_detrending(method[0], order=method[1], method_idx=method_idx)
                    elif method[0] == 'custom':
                        lightcurves[i].set_detrending(method[0], function=method[1], method_idx=method_idx)

                    # Now set up the required parameters to be fitted
                    n_coeffs = lightcurves[i].n_detrending_params

                    for coeff_i in range(n_coeffs):
                        # Loop over each coefficient!
                        coeff_name = 'd{}_{}'.format(coeff_i, method_idx)

                        # Set up the limits of each fitting parameter
                        if limits is None:
                            # Limits are not provided.
                            # We hard-code a default of ±10
                            low_lim = -10
                            high_lim = 10

                        elif isinstance(limits[method_idx][0], Iterable):
                            # The limits for each parameter have been set
                            # individually
                            low_lim = limits[method_idx][coeff_i][0]
                            high_lim = limits[method_idx][coeff_i][1]

                        else:
                            # [low, high] limit provided to apply universally
                            # to all the coefs for this method.
                            low_lim = limits[method_idx][0]
                            high_lim = limits[method_idx][1]

                        if method[0] == 'custom':
                            # Work out telescope, filter, epoch dependencies
                            tel_dep = coeff_i + 1 in method[2]
                            filt_dep = coeff_i + 1 in method[3]
                            epoch_dep = coeff_i + 1 in method[4]
                        else:
                            # We assume that the detrending coefficients are
                            # independent for all lightcurves
                            # (i.e. all dependencies are True)
                            tel_dep = True
                            filt_dep = True
                            epoch_dep = True

                        # Change the indices to None if not dependent
                        if tel_dep:
                            coeff_telescope_idx = telescope_idx
                        else:
                            coeff_telescope_idx = None
                        if  filt_dep:
                            coeff_filter_idx = filter_idx
                        else:
                            coeff_filter_idx = None
                        if epoch_dep:
                            coeff_epoch_idx = epoch_idx
                        else:
                            coeff_epoch_idx = None

                        if coeff_name not in self.priors:
                            # Need to initialise the entry in the priors dict
                            self.detrending_coeffs[method_idx].append(coeff_name)

                            # Have to deal with nth order and custom separately
                            if method[0] == 'nth order':
                                shape = (self.n_telescopes, self.n_filters, self.n_epochs)

                                # Initialise the prior
                                self.priors[coeff_name] = ParamArray(coeff_name, shape, True, True, True)

                            elif method[0] == 'custom':
                                # Work out the shape needed for the ParamArray
                                shape = ([1, self.n_telescopes][tel_dep],
                                         [1, self.n_filters][filt_dep],
                                         [1, self.n_epochs][epoch_dep])

                                # Initialise the prior
                                self.priors[coeff_name] = ParamArray(coeff_name, shape, tel_dep, filt_dep, epoch_dep)


                        # Now set up the fitting. We assume a uniform prior
                        # We have to check here for parameters which are
                        # telescope/filter/epoch dependent!
                        if coeff_name not in self.fitting_params[:,0]:
                            # The parameter has not been set up for fitting
                            # -> no checks needed
                            self.add_uniform_fit_param(coeff_name, low_lim, high_lim, coeff_telescope_idx, coeff_filter_idx, coeff_epoch_idx)


                        else:
                            # Run a check to see if the param is at least one
                            # of telescope, filter, or epoch dependent. If so,
                            # this needs to be fitted, but otherwise ignored.
                            if tel_dep or filt_dep or epoch_dep:
                                self.add_uniform_fit_param(coeff_name, low_lim, high_lim, coeff_telescope_idx, coeff_filter_idx, coeff_epoch_idx)

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

        # Store some useful information
        self.fit_ld = True
        self.ld_fit_method = fit_method
        self.filters = filters
        self._n_ld_samples = n_samples
        self._do_ld_mc = do_mc
        self._ld_cache_path = cache_path

        # Set up fitting for each LDC
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
        '''
        if self.normalise:
            raise ValueError('Detrending is already initialised. You need to make a new PriorInfo to use another detrending method!')

        for i in np.ndindex(lightcurves.shape):
            telescope_idx, filter_idx, epoch_idx = i

            if lightcurves[i] is not None:
                # A light curve exists. Set up normalisation
                best, low, high = lightcurves[i].set_normalisation()
                self.add_uniform_fit_param('norm', low, high, telescope_idx, filter_idx, epoch_idx)

        self.normalise = True

    ###############################################################
    #             CONVERSIONS FOR FITTING ROUTINES                #
    ###############################################################

    def _convert_unit_cube(self, cube):
        '''
        Takes the unit cube provided by dynesty (all values between 0 and 1)
        and converts them into physical values
        '''
        new_cube = np.zeros(len(self.fitting_params))

        for i, param_info in enumerate(self.fitting_params):
            name, tidx, fidx, eidx = param_info
            new_cube[i] = self.priors[name].from_unit_interval(cube[i], tidx, fidx, eidx)

        return new_cube

    def _interpret_param_array(self, array):
        '''
        Interprets the parameter cube generated by dynesty and returns them in
        a format usable by the LikelihoodCalculator
        '''

        # Initialse a dictionary to store the physical results in
        result = {}

        for name in self.priors.keys():
            # First we initialise the entry in the results dictionary
            result[name] = self.priors[name].generate_blank_ParamArray()

        initialised_params = []
        # Now we add in the results
        for i, param_info in enumerate(self.fitting_params):
            name, tidx, fidx, eidx = param_info
            result[name][tidx, fidx, eidx] = array[i]
            initialised_params.append(name)

        # If things aren't being fitted, add these in too
        for name in result.keys():
            if name not in initialised_params:
                for i in np.ndindex(self.priors[name].shape):
                    result[name][i] = self.priors[name][i]

        return result

    def _interpret_final_results(self, results):
        '''
        Generates a dictionary of results and a dictionary of errors from the
        final results of a run
        '''
        best_results = results.best
        best_result_errors = results.uncertainties

        result_dict = {}
        errors_dict = {}

        # Initialise the entries in the dictionaries
        for name in self.priors.keys():
            # First we initialise the entry in the results dictionary
            result_dict[name] = self.priors[name].generate_blank_ParamArray()
            errors_dict[name] = self.priors[name].generate_blank_ParamArray()

        initialised_params = []
        # Go through the results object and add in the results - only includes
        # fitted parameters
        for i, param_info in enumerate(self.fitting_params):
            name, tidx, fidx, eidx = param_info
            result_dict[name][tidx, fidx, eidx] = best_results[i]
            errors_dict[name][tidx, fidx, eidx] = best_result_errors[i]
            initialised_params.append(name)

        # Now go through and fill in any non-fitted parameters
        for name in result_dict.keys():
            if name not in initialised_params:
                for i in np.ndindex(self.priors[name].shape):
                    result_dict[name][i] = self.priors[name][i]
                    errors_dict[name][i] = 0

        return result_dict, errors_dict


    ###############################################################
    #                          MISC                               #
    ###############################################################
    def __str__(self):
        print_str = 'Priors:\n'
        print_str += 'Limb darkening model: {}\n'.format(self.limb_dark)
        print_str += 'n telescopes: {}\n'.format(self.n_telescopes)
        print_str += 'n filters: {}\n'.format(self.n_filters)
        print_str += 'n epochs: {}\n'.format(self.n_epochs)
        for var in self.priors:
            print_str += self.priors[var].__str__()
        print_str += 'Total {} fitting parameters'.format(len(self.fitting_params))
        return print_str
