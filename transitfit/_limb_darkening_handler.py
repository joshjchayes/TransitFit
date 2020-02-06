'''
Class to handle limb darkening

'''
import numpy as np
from ._ldtk_handler import LDTKHandler

class LimbDarkeningHandler:
    def __init__(self, default_model, low_lim=-5, high_lim=5):
        '''
        The LimbDarkeningHandler is designed to convert fitting parameters in
        the range [0,1] and convert them into physically allowed values for
        limb darkening coefficients for different models. This conversion is
        based on Kipping 2013 https://arxiv.org/abs/1308.0009.

        Parameters
        ----------
        default_model : str
            The default limb darkening model to use. Accepted values are
                - 'linear'
                - 'quadratic'
                - 'squareroot'
                - 'power2'
                - 'nonlinear'
            This model will be the default for conversions unless otheriwse
            specified.
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
        '''

        if default_model not in ['linear','quadratic','squareroot','power2','nonlinear']:
            raise ValueError('Unrecognised limb darkening model {}'.format(model))

        self.default_model = default_model

        # the default high and low bounds to use for unconstrained values
        self.high = high_lim
        self.low = low_lim

        self.ldtk_handler = None

    def get_required_coefficients(self, model=None):
        '''
        Finds the number of coefficients required for a given model and returns
        a string name for each of the form 'qX' where X is a number.

        Parameters
        ----------
        model : str or None, optional
            Model to get the coefficients for. Default is self.default_model

        Returns
        -------
        coeficient names : list of str
            The coefficient names
        '''
        if model is None:
            model = self.default_model

        if model == 'linear':
            return ['q0']
        if model in ['quadratic','squareroot', 'power2']:
            return ['q0', 'q1']
        if model == 'nonlinear':
            return ['q0', 'q1', 'q2', 'q3']

        raise ValueError('Unrecognised model {}'.format(model))

    def convert_qtoA(self, *q, model=None):
        '''
        Takes parameters q distributed between [0,1] and converts them into
        physical values for limb darkening coefficients.

        Conversions for quadratic, square root, and logarithmic are from
        Kipping 2013 https://arxiv.org/abs/1308.0009 for two-parameter limb
        darkening methods

        Notes
        -----
        This is the inverse of convert_Atoq
        '''
        if model is None:
            model = self.default_model

        if model == 'linear':
            # coefficient is limited to 0<A<1 by Kipping criteria
            return q[0]

        if model == 'quadratic':
            return 2 * np.sqrt(q[0]) * q[1], np.sqrt(q[0]) * (1 - 2 * q[1])

        if model == 'squareroot':
            return np.sqrt(q[0]) * (1 - 2 * q[1]), 2 * np.sqrt(q[0]) * q[1]

        if model == 'power2':
            # This parameterisation has been derived for TransitFit and is
            # explained in the paper
            u1 = q[0] * (1 - self.low) + self.low

            if u1 < 0:
                # negative quadrant
                return u1, self.low * (1 - q[1])

            else:
                # positive quadrant
                return u1, self.high * q[1]

        if model == 'nonlinear':
            # This is an 'outstanding and formidable problem' to apply the
            # Kipping (2013) method to (see Kipping 2016) and as such, we can
            # only fit for free parameters, without reparameterising to allow
            # only physically valid combinations.
            A = q[0] * (self.high - self.low) + self.low
            B = q[1] * (self.high - self.low) + self.low
            C = q[2] * (self.high - self.low) + self.low
            D = q[3] * (self.high - self.low) + self.low

            return A, B, C, D

        raise ValueError('Unrecognised model {}'.format(model))

    def convert_Atoq(self, *A, model=None):
        '''
        Takes actual values of the LD coefficients and converts them to a
        value q between [0,1].

        Conversions for quadratic, square root, and logarithmic are from
        Kipping 2013 https://arxiv.org/abs/1308.0009 for two-parameter limb
        darkening methods

        Notes
        -----
        This is the inverse of convert_qtoA
        '''
        if model is None:
            model = self.default_model

        if model == 'linear':
            # coefficient is limited to 0<A<1 by Kipping criteria
            return q[0]

        if model == 'quadratic':
            return (A[0] + A[1]) ** 2, A[0]/(2 * (A[0] + A[1]))

        if model == 'squareroot':
            return (A[0] + A[1]) ** 2, A[1]/(2 * (A[0] + A[1]))

        if model == 'power2':
            # This parameterisation has been derived for TransitFit and is
            # explained in the paper
            q1 = (A[0] - self.low)/(1 - self.low)

            if A[0] < 0:
                # negative quadrant
                return u1, 1 - (A[1]/self.low)

            else:
                # positive quadrant
                return u1, A[1]/self.high

        if model == 'nonlinear':
            # This is an 'outstanding and formidable problem' to apply the
            # Kipping (2013) method to (see Kipping 2016) and as such, we can
            # only fit for free parameters, without reparameterising to allow
            # only physically valid combinations.
            q0 = (A[0] - self.low) /  (self.high - self.low)
            q1 = (A[1] - self.low) /  (self.high - self.low)
            q2 = (A[2] - self.low) /  (self.high - self.low)
            q3 = (A[3] - self.low) /  (self.high - self.low)

            return q0, q1, q2, q3

        raise ValueError('Unrecognised model {}'.format(model))

    def initialise_ldtk(self, host_T, host_logg, host_z, filters,
                        model=None, n_samples=20000, do_mc=False,
                        cache_path=None):
        '''
        Sets up an LDTKHandler to deal with interfacing between ldtk and
        TransitFit

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
        ld_method : str or None, optional
            The model of limb darkening to use. Allowed values are 'linear',
            'quadratic', 'squareroot', 'power2', and 'nonlinear'. If None, will
            default to self.default_model. Default is None.
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
        if model is None:
            model = self.default_model

        self.ldtk_handler = LDTKHandler(host_T, host_logg, host_z, filters,
                                        model, n_samples, do_mc, cache_path)


    def ldtk_lnlike(self, coeffs, model=None):
        '''
        Uses LDTK to evaluate the log likelihood for a set of coefficients.
        Note that coeffs should be the values returned by convert_qtoA
        and NOT the unit values.

        Parameters
        ----------
        coeffs : array_like, shape (n_filters, n_coeffs)
            The coefficients to evaluate the log likelihood for.
        ld_model : str, optional
            The model to use. Defaults to self.default_model

        Returns
        -------
        lnlike : float
            The log likelihood of the coefficients
        '''

        if self.ldtk_handler is None:
            raise Exception('ldtk_handler is not initialised. Please initialise using initialise_ldtk()')

        if model is None:
            model = self.default_model

        return self.ldtk_handler.lnlike(coeffs, model)

    def ldtk_estimate(self, ld0_values, model=None):
        '''
        Uses LDTK to estimate the LDCs for all filters when given a set of LDCs
        for filter 0, based on the ratios between the best values
        found in initialisation.

        Parameters
        ----------
        ld1_values : float or array_like
            The LD parameters for filter 0
        ld_model : str, optional
            The limb darkening model to use. Defaults to self.default_model

        Returns
        -------
        all_ld_values : array_like, shape (n_filters, n_coeffs)
            The estimated limb darkening parameters
        '''
        if self.ldtk_handler is None:
            raise Exception('ldtk_handler is not initialised. Please initialise using initialise_ldtk()')

        if model is None:
            model = self.default_model

        return self.ldtk_handler.estimate_values(ld0_values, model)
