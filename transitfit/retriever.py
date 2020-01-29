'''
TransitFit is a module to fit transit light curves using BATMAN

This is an update to Retriever which couples the rp and t0 for
same wavelength and epoch respectively
'''
import numpy as np
from ._likelihood import LikelihoodCalculator
from ._utils import validate_data_format, get_normalised_weights, get_covariance_matrix
from .io import save_results, print_results, save_final_light_curves
from .plotting import plot_individual_lightcurves
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
        pass

    def run_dynesty(self, times, depths, errors, priorinfo,
                    maxiter=None, maxcall=None,
                    nlive=300, plot=True, dlogz=None, savefname='outputs.csv',
                    output_folder='./final_light_curves', plot_folder='./plots',
                    figsize=(12,8), plot_color='dimgrey', plot_titles=None,
                    add_plot_titles=True, plot_fnames=None, **dynesty_kwargs):
        '''
        Runs a dynesty retrieval on the given data set

        Parameters
        ----------
        times : array_like
            The times of the transit data. Should be in BJD.
        depths : array_like
            The flux measurements taken of the target star, normalised to a
            baseline of 1.
        errors : array_like
            The errors associated with the depths.
        priorinfo : PriorInfo
            The priors for fitting. This basically sets the intervals over
            which each parameter is fitted.
        maxiter : int or None, optional
            The maximum number of iterations to run. If None, will
            continue until stopping criterion is reached. Default is None.
        maxcall : int or None, optional
            The maximum number of likelihood calls in retrieval. If None, will
            continue until stopping criterion is reached. Default is None.
        nlive : int, optional
            The number of live points in the nested retrieval. Default is 300
        dlogz : float, optional
            Iteration will stop when the estimated contribution of the
            remaining prior volume to the total evidence falls below this
            threshold. Explicitly, the stopping criterion is
            `ln(z + z_est) - ln(z) < dlogz`, where z is the current evidence
            from all saved samples and z_est is the estimated contribution from
            the remaining volume. The default is `1e-3 * (nlive - 1) + 0.01`.
        output_folder : str, optional
            The folder to save the files to. Default is './final_light_curves'
        plot : bool, optional
            If True, will plot the data and the best fit model on a Figure.
            Default is True
        plot_folder : str, optional
            Path to folder to save plots to. Default is './plots'
        figsize : tuple, optional
            The fig size for each plotted figure. Default is (12, 8)
        plot_color : matplotlib color, optional
            The base color for plots. Default is 'dimgray'
        plot_titles : None or array_like, shape (n_filters, n_epochs), optional
            The titles to use for each plot. If None, will default to
            'Filter X Epoch Y'. Default is None.
        add_plot_titles : bool, optional
            If True, will add titles to plots. Default is True.
        plot_fnames : None or array_like, shape (n_filters, n_epochs), optional
            The file names to use for each plot. If None, will default to
            'fX_eY.pdf'. Default is None.
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

        n_dof = 0.
        for row in times:
            for c in row:
                if c is not None:
                    n_dof += len(c)

        # Make an object to calculate all the likelihoods
        likelihood_calc = LikelihoodCalculator(times, depths, errors, priorinfo)

        def dynesty_transform_prior(cube):
            '''
            Convert the unit cube values to a parameter value using a uniform
            distribution bounded by the values given by the PriorInfo
            '''
            return priorinfo._convert_unit_cube(cube)


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
                                                            params['norm'],
                                                            detr_func,
                                                            d)

            if priorinfo.fit_ld and not priorinfo.ld_fit_method == 'independent':
                return ln_likelihood + priorinfo.ld_handler.ldtk_lnlike(np.array(u).T, limb_dark)
            else:
                return ln_likelihood

        # Set up and run the sampler here!!
        sampler = NestedSampler(dynesty_lnlike, dynesty_transform_prior,
                                ndims, bound='multi', sample='rwalk',
                                update_interval=float(ndims), nlive=nlive,
                                **dynesty_kwargs)
        sampler.run_nested(maxiter=maxiter, maxcall=maxcall, dlogz=dlogz)

        results = sampler.results

        # Get some normalised weights
        results.weights = get_normalised_weights(results)

        # Calculate a covariance matrix for these results to get uncertainties
        cov = get_covariance_matrix(results)

        # Get the uncertainties from the diagonal of the covariance matrix
        diagonal = np.diag(cov)

        uncertainties = np.sqrt(diagonal)

        # Add the covariance matrix and uncertainties to the results object
        results.cov = cov
        results.uncertainties = uncertainties

        # Print the Results
        print_results(results, priorinfo, n_dof)

        # Save to outputs?
        try:
            save_results(results, priorinfo, savefname)
        except Exception as e:
            print('The following exception was raised  whilst saving parameter results. I have just returned the results dictionary')
            print(e)

        try:
            save_final_light_curves(times, depths, errors, priorinfo,
                                    results, output_folder)
        except Exception as e:
            print('The following exception was raised whilst saving final light curves. I have just returned the results dictionary')
            print(e)

        if plot:
            try:
                plot_individual_lightcurves(times, depths, errors, priorinfo,
                                            results, folder_path=plot_folder,
                                            color=plot_color, titles=plot_titles, add_titles=add_plot_titles,
                                            fnames=plot_fnames)
            except Exception as e:
                # TODO: Try plotting from files rather than results objects
                print('Plotting error: I have failed to plot anything due to the following error:')
                print(e)
                raise


        return results
