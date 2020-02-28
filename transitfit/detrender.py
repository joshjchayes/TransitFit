'''
detrender.py

A module to deal with arbitrary detrending

'''

import numpy as np
from inspect import signature
import inspect

class DetrendingFunction:
    def __init__(self, function):
        '''
        The DetrendingFunction is designed to handle detrending with an arbitrary
        function.

        Basically gets around dealing with varying number of args in a neat way
        '''
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

        self.n_required_args = args
        self.n_kwargs = kwargs
        self.n_params = len(params)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
