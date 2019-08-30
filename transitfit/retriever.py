'''
TransitFit is a module to fit transit light curves using BATMAN
'''


import numpy as np
from transitfit import LikelihoodCalculator
from dynesty import NestedSampler
import matplotlib.pyplot as plt
import batman

class Retriever:
    def __init__(self):
        '''
        The basic retriever object for TransitFit.
        '''
        # The keys which are used for each of the parameters, in order.
        self._parameter_keys = ['P', 'a', 'inc', 't0', 'rp']

    def run_dynesty(self, times, depths, errors, priorinfo, eccentricity=0,
                    w=90, limb_dark='quadratic', limb_dark_params=[0.1, 0.3],
                    maxiter=None, maxcall=None, nlive=100, plot_best=True,
                    **dynesty_kwargs):
        '''
        Runs a dynesty retrieval on the given data set

        '''
        # Make an object to calculate all the likelihoods
        likelihood_calc = LikelihoodCalculator(times, depths, errors)

        def dynesty_transform_prior(cube):
            '''
            Convert the unit cube values to a parameter value using a uniform
            distribution bounded by the values given by the PriorInfo
            '''
            new_cube = np.zeros(len(cube))
            for i in range(len(cube)):
                new_cube[i] = priorinfo._value_from_unit_interval(cube[i], self._parameter_keys[i])
            return new_cube

        def dynesty_lnlike(cube):
            '''
            Function to pass to dynesty sampler to calculate lnlikelihood
            '''
            ln_likelihood = likelihood_calc.find_likelihood(cube[3], cube[0],
                                                            cube[4], cube[1],
                                                            cube[2], eccentricity,
                                                            w, limb_dark, limb_dark_params)
            return ln_likelihood

        # Set up and run the sampler here!!
        sampler = NestedSampler(dynesty_lnlike, dynesty_transform_prior,
                                len(self._parameter_keys), bound='multi',
                                sample='rwalk', update_interval=float(len(self._parameter_keys)),
                                nlive=nlive, **dynesty_kwargs)
        sampler.run_nested(maxiter=maxiter, maxcall=maxcall)

        results = sampler.results
        # Save to outputs?

        best_results = results.samples[np.argmax(results.logl)]

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

        return results
