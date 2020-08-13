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

