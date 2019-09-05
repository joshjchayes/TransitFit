'''
_params

A class for parameters in TransitFit which can be retrieved. These are used
by the PriorInfo to determine dimensionality etc.

'''


class _Param:
    def __init__(self, best, low_lim=None, high_lim=None):
        self.default_value = best
        self.low_lim = low_lim
        self.high_lim = high_lim

    def from_unit_interval(self, u):
        raise NotImplementedError


class _UniformParam(_Param):
    def __init__(self, best, low_lim, high_lim):
        # TODO: sanity checks on values and low < best < high

        super().__init__(best, low_lim, high_lim)



    def from_unit_interval(self, u):
        '''
        Function to convert value u in range (0,1], will convert to a value to
        be used by Batman
        '''
        return u * (self.high_lim - self.low_lim) + self.low_lim
