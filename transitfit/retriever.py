'''
TransitFit is a module to fit transit light curves using BATMAN

This is an update to Retriever which couples the rp and t0 for
same wavelength and epoch respectively
'''
import numpy as np
from ._likelihood import LikelihoodCalculator
from ._utils import validate_data_format
from .io import save_results
from .plotting import plot_best
from dynesty import NestedSampler
import dynesty.utils
import matplotlib.pyplot as plt
import batman
import csv

class Retriever:
    def __init__(self):
        '''
        The basic retriever object for TransitFit. This does all of the
        heavy lifting of running retrieval through the run_dynesty() function
        '''
        # The keys which are used for each of the parameters, in order.
        pass

    def run_dynesty(self, times, depths, errors, priorinfo,
                    limb_dark='quadratic', maxiter=None, maxcall=None,
                    nlive=300, plot=True, savefname='outputs.csv',
                    **dynesty_kwargs):
        '''
        Runs a dynesty retrieval on the given data set

        Parameters
        ----------
        times : array_like
            The times of the transit data
        depths : array_like
            The flux measurements taken of the target star, normalised to a
            baseline of 1.
        errors : array_like
            The errors associated with the depths.
        priorinfo : PriorInfo
            The priors for fitting. This basically sets the intervals over
            which each parameter is fitted.
        limb_dark : str, optional
            The limb darkening model to use. Default is 'quadratic'
        maxiter : int or None, optional
            The maximum number of iterations to run. If None, will
            continue until stopping criterion is reached. Default is None.
        maxcall : int or None, optional
            The maximum number of likelihood calls in retrieval. If None, will
            continue until stopping criterion is reached. Default is None.
        nlive : int, optional
            The number of live points in the nested retrieval. Default is 100
        plot_best : bool, optional
            If True, will plot the data and the best fit model on a Figure.
            Default is True
        **dynesty_kwargs : optional
            Additional kwargs to pass to dynesty.NestedSampler

        Returns
        -------
        results
            The dynesty results dictionary
        best : dict
            A dictionary containing the best values for each parameter.
        '''
        # number of dimensions being fitted.
        ndims = len(priorinfo.fitting_params)

        times, depths, errors = validate_data_format(times, depths, errors)

        # Make an object to calculate all the likelihoods
        likelihood_calc = LikelihoodCalculator(times, depths, errors)

        def dynesty_transform_prior(cube):
            '''
            Convert the unit cube values to a parameter value using a uniform
            distribution bounded by the values given by the PriorInfo
            '''
            new_cube = np.zeros(len(cube))
            for i in range(len(cube)):
                new_cube[i] = priorinfo._from_unit_interval(i, cube[i])
            return new_cube

        def dynesty_lnlike(cube):
            '''
            Function to pass to dynesty sampler to calculate lnlikelihood
            '''

            params = priorinfo._interpret_param_array(cube)

            # Get the limb darkening details and coefficient values
            limb_dark = priorinfo.limb_dark
            u = [params[key] for key in priorinfo.limb_dark_coeffs]


            if priorinfo.detrend:
                # Do the detrending things
                detr_func = priorinfo.detrending_function
                d = np.array([params[key] for key in priorinfo.detrending_coeffs])

            else:
                # Don't detrend
                detr_func = None
                d = None


            ln_likelihood = likelihood_calc.find_likelihood(params['t0'],
                                                            params['P'],
                                                            params['rp'],
                                                            params['a'],
                                                            params['inc'],
                                                            params['ecc'],
                                                            params['w'],
                                                            limb_dark,
                                                            np.array(u).T,
                                                            detr_func,
                                                            d)

            return ln_likelihood

        # Set up and run the sampler here!!
        sampler = NestedSampler(dynesty_lnlike, dynesty_transform_prior,
                                ndims, bound='multi', sample='rwalk',
                                update_interval=float(ndims), nlive=nlive,
                                **dynesty_kwargs)
        sampler.run_nested(maxiter=maxiter, maxcall=maxcall)

        results = sampler.results

        # Work out some weights so that we can calculate errors
        normalized_weights = np.exp(results.logwt - results.logwt.max())/np.sum(np.exp(results.logwt - results.logwt.max()))
        results.weights = normalized_weights



        # Save to outputs?
        try:
            save_results(results, priorinfo, savefname)
            best_results = results.samples[np.argmax(results.logl)]
        except Exception as e:
            print(e)
            print('Exception raised whilst saving results. I have just returned the results dictionary')

        if plot:
            plot_best(times, depths, errors, priorinfo, results)

        return results
