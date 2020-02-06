'''
Class to handle limb darkening parameters through PyLDTK

'''

import numpy as np
from ldtk import LDPSetCreator, BoxcarFilter
import os

_implemented_ld_models = ['linear', 'quadratic', 'nonlinear', 'power2', 'squareroot']

class LDTKHandler:
    def __init__(self, host_T, host_logg, host_z, filters,
                 ld_model='quadratic', n_samples=20000, do_mc=False,
                 cache_path=None):
        '''
        The LDTKHandler provides an easy way to interface ldtk with TransitFit.

        Parameters
        ----------
        host_T : tuple
            The effective temperature of the host star, in Kelvin, given as a
            (value, uncertainty) pair.
        host_logg : tuple
            The log_10 of the surface gravity of the host star, with gravity
            measured in cm/s2. Should be given as a (value, uncertainty) pair.
        host_z : tuple
            The metalicity of the host, given as a (value, uncertainty) pair.
        filters : array_like
            The set of filters, given in [low, high] limits for the wavelengths
            with the wavelengths given in nanometers. The ordering of the
            filters should correspond to the filter_idx parameter used
            elsewhere.
        ld_method : str, optional
            The model of limb darkening to use. Allowed values are 'linear',
            'quadratic', 'squareroot', 'power2', and 'nonlinear'. Default is
            'quadratic'.
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
        if not ld_model in _implemented_ld_models:
            raise ValueError('Unrecognised ld_model {}'.format(ld_model))

        self.default_model = ld_model

        # Set up the filters
        print('Setting up filters')
        ldtk_filters = []
        for i, f in enumerate(filters):
            ldtk_filters.append(BoxcarFilter('{}'.format(i), f[0], f[1]))

        # Make the set creator, downloading data files if required
        if cache_path is not None:
            os.makedirs(cache_path, exist_ok=True)
        print('Making LD parameter set creator.')
        print('This may take some time as we may need to download files...')
        set_creator = LDPSetCreator(teff=host_T, logg=host_logg, z=host_z,
                                    filters=ldtk_filters, cache=cache_path)

        # Get the LD profiles from the set creator
        print('Obtaining LD profiles')
        self.profile_set = set_creator.create_profiles(nsamples=n_samples)

        # Find the 'best values' for each filter and then find the ratios
        # compared to the first.
        print('Finding coefficients and ratios')
        self.coeffs = {}
        self.ratios = {}
        for model in _implemented_ld_models:
            self.coeffs[model] = self._extract_best_coeffs(model)
            self.ratios[model] = self.coeffs[model][0] / self.coeffs[model][0][0]


    def estimate_values(self, ld0_values, ld_model):
        '''
        If given a set of LD param values for filter 0, will estimate the LD
        parameters for all filters based on the ratios between the best values
        found in initialisation.

        Parameters
        ----------
        ld1_values : float or array_like
            The LD parameters for filter 0
        ld_model : str
            The limb darkening model to use

        Returns
        -------
        all_ld_values : array_like, shape (n_filters, n_coeffs)
            The estimated limb darkening parameters
        '''

        return ld0_values * self.ratios[ld_model]

    def _extract_best_coeffs(self, ld_model):
        '''
        Extracts the best values for a limb darkening model for the filters

        Parameters
        ----------
        ld_model : str
            The limb darkening model to obtain the values for.

        Returns
        -------
        coeffs : array_like, shape (n_filters, n_coeffs)
            The coefficients for each filter
        err : array_like, shape (n_filters, n_coeffs)
            The uncertainty on each of the coefficients
        '''
        if ld_model == 'linear':
            coeff, err = self.profile_set.coeffs_ln()
        elif ld_model == 'quadratic':
            coeff, err = self.profile_set.coeffs_qd()
        elif ld_model == 'nonlinear':
            coeff, err = self.profile_set.coeffs_nl()
        elif ld_model == 'power2':
            coeff, err = self.profile_set.coeffs_p2()
        elif ld_model == 'squareroot':
            coeff, err = self.profile_set.coeffs_sq()

        else:
            raise ValueError('Unrecognised ld_model {}'.format(ld_model))

        return coeff, err


    def lnlike(self, coeffs, ld_model):
        '''
        Evaluates the log likelihood for a set of coefficients

        Parameters
        ----------
        coeffs : array_like, shape (n_filters, n_coeffs)
            The coefficients to evaluate the log likelihood for.
        ld_model : str, optional
            The model to use. Defaults to self.default_model

        '''
        if ld_model == 'linear':
            return self.profile_set.lnlike_ln(coeffs)
        if ld_model == 'quadratic':
            return self.profile_set.lnlike_qd(coeffs)
        if ld_model == 'nonlinear':
            return self.profile_set.lnlike_nl(coeffs)
        if ld_model == 'power2':
            return self.profile_set.lnlike_p2(coeffs)
        if ld_model == 'squareroot':
            return self.profile_set.lnlike_sq(coeffs)

        raise ValueError('Unrecognised ld_model {}'.format(ld_model))