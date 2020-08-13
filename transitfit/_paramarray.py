'''
_param_array.py

'''
import numpy as np
from ._params import _Param, _UniformParam, _GaussianParam


class ParamArray:
    def __init__(self, name, shape, telescope_dependent, filter_dependent,
                 epoch_dependent, default_value=None, lightcurves=None):
        '''
        The _ParamArray is designed to handle parameters which are being fitted
        by TransitFit. It can deal with parameters which are being fitted over
        different combinations of telescope, filter, and epoch

        If a parameter is lightcurve specific, providing lightcurves will
        ensure that only parameters where a lightcurve exists will be
        initialised
        '''
        # Store some useful data
        self.name = name
        self.shape = shape

        self.telescope_dependent = telescope_dependent
        self.filter_dependent = filter_dependent
        self.epoch_dependent = epoch_dependent
        self.global_param = sum((self.telescope_dependent,
                                self.filter_dependent, self.epoch_dependent)) == 0
        self.default_value = default_value
        self.lightcurves = lightcurves

        # Check that shape matches up with the given dependencies
        if not self.telescope_dependent and not shape[0] == 1:
            raise ValueError('The ParamArray "{}" is not telescope dependent, but the shape {} implies that it is!'.format(self. name, self.shape))
        if not self.filter_dependent and not shape[1] == 1:
            raise ValueError('The ParamArray "{}" is not filter dependent, but the shape {} implies that it is!'.format(self. name, self.shape))
        if not self.epoch_dependent and not shape[2] == 1:
            raise ValueError('The ParamArray "{}" is not epoch dependent, but the shape {} implies that it is!'.format(self. name, self.shape))

        # Create the array which we will fill with _Params
        if lightcurves is None:
            # No lightcurves provided, we can just fill the array with params
            self.array = np.full(shape, default_value, object)

        else:
            if not lightcurves.shape == shape:
                raise ValueError('shape and lightcurves.shape do not match')

            # Only fill those where the lightcurves are
            self.array = np.full(shape, None, object)
            for i in np.ndindex(lightcurves.shape):
                if lightcurves[i] is not None:
                    self.array[i] = default_value

    def set_value(self, value, telescope_idx=None, filter_idx=None,
                  epoch_idx=None):
        '''
        Sets the array entry at the provided value

        Value can be any object
        '''
        idx = self._generate_idx(telescope_idx, filter_idx, epoch_idx)
        self.array[idx] = value

    def get_value(self, telescope_idx=None, filter_idx=None,
                  epoch_idx=None):
        idx = self._generate_idx(telescope_idx, filter_idx, epoch_idx)
        return self.array[idx]

    def add_uniform_fit_param(self, low_lim, high_lim, telescope_idx=None,
                              filter_idx=None, epoch_idx=None):
        '''
        Adds a parameter to be fitted with uniform sampling
        '''
        self.set_value(_UniformParam(low_lim, high_lim),
                       telescope_idx, filter_idx, epoch_idx)

    def add_gaussian_fit_param(self, mean, stdev, telescope_idx=None,
                               filter_idx=None, epoch_idx=None):
        '''
        Adds a gaussian sampled fitting parameter
        '''
        self.set_value(_GaussianParam(mean, stdev),
                       telescope_idx, filter_idx, epoch_idx)

    def from_unit_interval(self, u, telescope_idx=None, filter_idx=None,
                           epoch_idx=None):
        '''
        Converts a value u in the range [0,1] to a physical value. Used in the
        dynesty routine
        '''
        idx = self._generate_idx(telescope_idx, filter_idx, epoch_idx)
        return self.array[idx].from_unit_interval(u)

    def generate_blank_ParamArray(self):
        '''
        Produces a ParamArray with the same telescope, filter, and epoch
        dependencies and shape, but with everything else initialised to None
        '''
        return ParamArray(self.name, self.shape, self.telescope_dependent,
                          self.filter_dependent, self.epoch_dependent)

    def _generate_idx(self, telescope_idx, filter_idx, epoch_idx):
        '''
        Takes the given indices and converts them into a usable index, omitting
        any that are given for variables that the ParamArray does not depend on
        and raises ValueErrors if None is provided when required.
        '''
        if self.global_param:
            return (0, 0, 0)

        idx = ()
        if self.telescope_dependent:
            if telescope_idx is None:
                raise ValueError('telescope_idx must be provided')
            idx += (telescope_idx,)
        else:
            idx += (0,)

        if self.filter_dependent:
            if filter_idx is None:
                raise ValueError('filter_idx must be provided')
            idx += (filter_idx,)
        else:
            idx += (0,)

        if self.epoch_dependent:
            if epoch_idx is None:
                raise ValueError('epoch_idx must be provided')
            idx += (epoch_idx,)
        else:
            idx += (0,)

        return idx

    def __getitem__(self, idx):
        telescope_idx, filter_idx, epoch_idx = idx
        idx = self._generate_idx(telescope_idx, filter_idx, epoch_idx)
        return self.array[idx]

    def __setitem__(self, idx, value):
        telescope_idx, filter_idx, epoch_idx = idx
        idx = self._generate_idx(telescope_idx, filter_idx, epoch_idx)
        self.array[idx] = value
