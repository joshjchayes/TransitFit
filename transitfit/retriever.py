'''
TransitFit is a module to fit transit light curves using BATMAN
'''


import numpy as np
from ._likelihood import LikelihoodCalculator
from dynesty import NestedSampler
import matplotlib.pyplot as plt
import batman

class Retriever:
    def __init__(self):
        '''
        The basic retriever object for TransitFit. This does all of the
        heavy lifting of running retrieval through the run_dynesty() function
        '''
        # The keys which are used for each of the parameters, in order.
        pass

    def run_dynesty(self, times, depths, errors, priorinfo,
                    maxiter=None, maxcall=None, nlive=100, plot_best=True,
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
        eccentricity : float, optional
            Eccentricity of the orbit. Default is 0.
        w : float, optional
            longitude of periastron (in degrees). Default is 90
        limb_dark : str, optional
            The limb darkening model to use. Default is 'quadratic'
        limb_dark_params : array_like, optional
            The constants for the limb darkening model. Default is [0.1, 0.3].
        maxiter : int or None, optional
            The maximum number of iterations to run. If None, will
            continue until stopping criterion is reached. Default is None.
        maxcall : int or None, optional
            The maximum number of likelihood calls in retrieval. If None, will
            continue until stopping criterion is reached. Default is None.
        nlive : int, optional
            The number of live points in the nested retrieval. Default is 100
        plot_best : bool, optional
            If True, will plot the data and the best fir model on a Figure.
            Default is True
        **dynesty_kwargs : optional
            Additional kwargs to pass to dynesty.NestedSampler

        Returns
        -------
        results
        '''
        # number of dimensions being fitted.
        ndims = len(priorinfo.fitting_params)

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


            ln_likelihood = likelihood_calc.find_likelihood(params['t0'],
                                                            params['P'],
                                                            params['rp'],
                                                            params['a'],
                                                            params['inc'],
                                                            params['ecc'],
                                                            params['w'],
                                                            params['limb_dark'],
                                                            params['limb_dark_params'])

            return ln_likelihood

        # Set up and run the sampler here!!
        sampler = NestedSampler(dynesty_lnlike, dynesty_transform_prior,
                                ndims, bound='multi', sample='rwalk',
                                update_interval=float(ndims), nlive=nlive,
                                **dynesty_kwargs)
        sampler.run_nested(maxiter=maxiter, maxcall=maxcall)

        results = sampler.results
        # Save to outputs?

        best_results = results.samples[np.argmax(results.logl)]

        '''
        if plot_best:
            print('Plotting best fit curve')
            likelihood_calc.update_params(best_results[3], best_results[0],
                                          best_results[4], best_results[1],
                                          best_results[2], eccentricity,
                                          w, limb_dark, limb_dark_params)
            model = batman.TransitModel(likelihood_calc.batman_params, times)
            light_curve = model.light_curve(likelihood_calc.batman_params)

            fig, ax = plt.subplots(1,1)
            ax.errorbar(times, depths, errors, alpha=0.8, fmt='xk', zorder=0,
            markersize=0.5)
            ax.plot(times, light_curve, linewidth=3)

            plt.show()
        '''
        return results
