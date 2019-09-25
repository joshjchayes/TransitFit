'''
PriorInfo objects

Object to handle and deal with prior info for retrieval
'''

import numpy as np
from ._params import _Param, _UniformParam
from .detrending_funcs import linear, quadratic, sinusoidal
from .detrender import DetrendingFunction

_prior_info_defaults = {'P':1, 'a':10, 'inc':90, 'rp':0.05, 't0':0, 'ecc':0,
                        'w':90, 'limb_dark':'quadratic', 'u1':0.1, 'u2':0.3,
                        'u3':-0.1, 'u4':0, 'num_wavelengths':1,
                        'num_times':1}

_detrending_functions = {'linear' : linear, 'quadratic' : quadratic,
                         'sinusoidal' : sinusoidal}

def setup_priors(P, a, inc, rp, t0, ecc=0, w=90, limb_dark='quadratic',
                 u1=0.1, u2=0.3, u3=None, u4=None, num_wavelengths=1,
                 num_times=1):
    '''
    Creates a PriorInfo object with some default inputs

    num_light_curves : int
        The number of light curves which will be fitted using the priors.
        Required to ensure that the rp and t0 priors are the right shapes
    '''
    default_priors = {}

    default_priors['num_wavelengths'] = num_wavelengths
    default_priors['num_times'] = num_times
    default_priors['P'] = P
    default_priors['a'] = a
    default_priors['inc'] = inc
    default_priors['rp'] = rp
    default_priors['t0'] = t0
    default_priors['ecc'] = ecc
    default_priors['w'] = w

    if  (limb_dark == "uniform" and len(limb_dark_params) != 0) or \
        (limb_dark == "linear" and len(limb_dark_params) != 1) or \
        (limb_dark == "quadratic" and len(limb_dark_params) != 2) or \
        (limb_dark == "logarithmic" and len(limb_dark_params) != 2) or \
        (limb_dark == "exponential" and len(limb_dark_params) != 2) or \
        (limb_dark == "squareroot" and len(limb_dark_params) != 2) or \
        (limb_dark == "power2" and len(limb_dark_params) != 2) or \
        (limb_dark == "nonlinear" and len(limb_dark_params) != 4):
        raise Exception("Incorrect number of coefficients for " + limb_dark + \
         " limb darkening; u should have the form:\n \
         u = [] for uniform LD\n \
         u = [u1] for linear LD\n \
         u = [u1, u2] for quadratic, logarithmic, exponential, squareroot, and power2 LD\n \
         u = [u1, u2, u3, u4] for nonlinear LD, or\n \
         u = [u1, ..., un] for custom LD")

     # Set up the limb darkening fitting
    default_priors['limb_dark'] = limb_dark
    if not limb_dark == 'uniform':
        limb_dark_params['u1'] = np.any(table[0] == 'u1')
        if not limb_dark == 'linear':
            limb_dark_params['u2'] = np.any(table[0] == 'u2')
            if limb_dark == 'nonlinear':
                limb_dark_params['u3'] = np.any(table[0] == 'u3')
                limb_dark_params['u4'] = np.any(table[0] == 'u4')

    default_priors['limb_dark_params'] = limb_dark_params


    return PriorInfo(default_priors, warn=False)

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
            print('Warning: you appear to be making the PriorInfo directly rather than using setup_priors. Proceed with caution, or use setup_priors. You can mute this warning by using warn=False.')

        self.fitting_params = []
        self._filter_idx = []
        self._epoch_idx = []
        self.priors = {}
        self.num_times = default_dict['num_times']
        self.num_wavelengths = default_dict['num_wavelengths']
        self.limb_dark = default_dict['limb_dark']

        # Sort out the limb darkening coefficients
        if self.limb_dark == 'uniform':
            self.limb_dark_coeffs = []
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

        # Set up a dictionary containing the parameters
        for key in default_dict.keys():
            if key in ['rp'] + self.limb_dark_coeffs:
                self.priors[key] = [_Param(default_dict[key]) for i in range(self.num_wavelengths)]
            elif key =='t0':
                self.priors[key] = [_Param(default_dict[key]) for i in range(self.num_times)]
            elif key in ['num_times', 'num_wavelengths', 'limb_dark']:
                pass
            else:
                self.priors[key] = _Param(default_dict[key])

        # Initialse a bit for the detrending
        self.detrend = False
        self.detrending_coeffs = []

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

            elif key in self.detrending_coeffs:
                result[key][fidx, eidx] = array[i]

            else:
                result[key] = array[i]

        # Now we consider any parameters which aren't being fitted
        # Note that we don't need to consider detrending coeffs as they will
        # ONLY be present at all if we are detrending
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
                        self.add_uniform_fit_param(key, 0, -100, 100, filter_idx=j, epoch_idx=k)
