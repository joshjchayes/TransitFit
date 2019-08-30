'''
Tests for TransitFit

'''

import numpy as np
import batman
from .likelihood import LikelihoodCalculator
from .priorinfo import PriorInfo
from .retriever import Retriever

def run_all_tests():
    '''
    Runs all tests
    '''
    test_LikelihoodCalculator()
    test_PriorInfo()

def make_dummy_lightcurve(error_size=0.01):
    '''Makes a dummy light curve with errors we can use'''
    # Set up the dummy data
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = 0.                        #time of inferior conjunction
    params.per = 1.                       #orbital period
    params.rp = 0.1                       #planet radius (in units of stellar radii)
    params.a = 15.                        #semi-major axis (in units of stellar radii)
    params.inc = 87.                      #orbital inclination (in degrees)
    params.ecc = 0.                       #eccentricity
    params.w = 90.                        #longitude of periastron (in degrees)
    params.limb_dark = "nonlinear"        #limb darkening model
    params.u = [0.5, 0.1, 0.1, -0.1]      #limb darkening coefficients [u1, u2, u3, u4]

    times = np.linspace(-0.025, 0.025, 1000)  #times at which to calculate light curve
    model = batman.TransitModel(params, times)    #initializes model
    dummy_data = model.light_curve(params)


    disp_arr = np.zeros(dummy_data.shape)
    errors = np.zeros(dummy_data.shape)
    for i, depth in enumerate(dummy_data):
        if not error_size == 0:
            disp = np.random.normal(0, error_size)
            disp_arr[i] = disp
            errors[i] = depth * error_size
        else:
            errors[i] = depth *  0.0001

    dummy_data += disp_arr

    return times, dummy_data, errors, params

def test_LikelihoodCalculator():
    '''Test that the LikelihoodCalculator actually produces some sane values'''

    # Set up the dummy data
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = 0.                        #time of inferior conjunction
    params.per = 1.                       #orbital period
    params.rp = 0.1                       #planet radius (in units of stellar radii)
    params.a = 15.                        #semi-major axis (in units of stellar radii)
    params.inc = 87.                      #orbital inclination (in degrees)
    params.ecc = 0.                       #eccentricity
    params.w = 90.                        #longitude of periastron (in degrees)
    params.limb_dark = "nonlinear"        #limb darkening model
    params.u = [0.5, 0.1, 0.1, -0.1]      #limb darkening coefficients [u1, u2, u3, u4]

    times = np.linspace(-0.025, 0.025, 1000)  #times at which to calculate light curve
    model = batman.TransitModel(params, times)    #initializes model
    dummy_data = model.light_curve(params)

    errors = dummy_data * 0.01  # Errors = 1%

    # Make a LikelihoodCalculator
    calculator = LikelihoodCalculator(times, dummy_data, errors)

    # Fit a perfect curve:
    l1 = calculator.find_likelihood(0, 1, 0.1, 15., 87, 0, 90, 'nonlinear', [0.5, 0.1, 0.1, -0.1])
    print('Likelihood of exact parameters: {}'.format(round(l1, 3)))

    # Fit a bad curve:
    l2 = calculator.find_likelihood(0, 1.1, 0.13, 12., 89, 0, 90, 'nonlinear', [0.5, 0.1, 0.1, -0.1])
    print('Likelihood of non-exact parameters: {}'.format(round(l2, 3)))


def test_PriorInfo():
    '''
    Tests that the PriorInfo can be made
    '''
    pi = PriorInfo()
    print('Prior info created')

def test_Retriever(error_size=0.001):
    '''
    Runs a test on the Retriever to make sure that it runs and returns sane
    results
    '''
    times, data, errors, params = make_dummy_lightcurve(error_size)

    retriever = Retriever()
    prior = PriorInfo([0.8, 1.2], [5, 20], t0=[-5, 5])

    result = retriever.run_dynesty(times, data, errors, prior)

    return result
