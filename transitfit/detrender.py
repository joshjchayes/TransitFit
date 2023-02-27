'''
detrender.py

A module to deal with arbitrary detrending

'''

import numpy as np
from inspect import signature
import inspect

class DetrendingFunction:
    '''
    The DetrendingFunction is designed to handle detrending with an arbitrary
    function.

    Basically gets around dealing with varying number of args in a neat way

    Parameters
    ----------
    function : function
        The detrending function. Must have signature of f(times, a, b, c,...,t0,P)
        where a, b, c etc are the detrending coefficients, t0 is the time of 
        conjunction and P is period.
    '''
    def __init__(self, function,method=None):
          

        # Store the function
        self.function = function

        # Do some analysis of the signature to get info on arg, kwargs and the like
        sig = signature(self.function)
        params = sig.parameters

        args = 0
        kwargs = 0
        for p in params:
            if params[p].default == inspect._empty:
                # No default value given - must be supplied
                args += 1
            else:
                kwargs += 1
        if method == 'custom':
            args=args-2 
            # For custom functions, 2 arguments are subtracted because 
            # they correpond to 't0' and 'P'

        self.n_required_args = args
        self.n_kwargs = kwargs
        
        # -1 assumes that the first entry is lightcurve, so won't be fitted
        try:
            self.n_params = function.order
        except:
            self.n_params = len(params)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
