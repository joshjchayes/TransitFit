'''
advanced_retriever

The advanced retriever is intended as the wrapper object for all processes in
TransitFit - i.e. it is the main thing that users will need to interact with.

This retriever is designed to handle large numbers of light curves, by
fitting individual filters, producing phase folded detrended light curves and
then fitting these across filters
'''

from .io import read_input_file, read_priors_file, parse_priors_list, read_filter_info, parse_filter_list
from ._likelihood import LikelihoodCalculator
from ._utils import get_normalised_weights, get_covariance_matrix
from . import io
from .plotting import plot_individual_lightcurves

import numpy as np
from dynesty import NestedSampler

class AdvancedRetriever:
    def __init__(self, data_files, priors, n_telescopes, n_filters, n_epochs,
                 filter_info=None, detrending_list=[['nth order', 1]],
                 limb_darkening_model='quadratic', host_T=None, host_logg=None,
                 host_z=None, ldtk_cache=None):
        '''
        The AdvancedRetriever handles all processes in TransitFit.

        The AdvancedRetriever is designed to handle large numbers of light
        curves, by fitting individual filters, producing phase folded detrended
        light curves and then fitting these across filters.
        '''
        # Save the basic input data
        self._data_input = data_files
        self._prior_input = priors
        self._filter_input = filter_info
        self.detrending_info = detrending_list
        self.limb_darkening_model = limb_darkening_model
        self.host_T = host_T
        self.host_logg = host_logg
        self.host_z = host_z
        self.ldtk_cache = ldtk_cache
        self.n_telescopes = n_telescopes
        self.n_filters = n_filters
        self.n_epochs = n_epochs

        # Read in the filters
        if type(self._filter_input) == str:
            self.filters = read_filter_info(self._filter_input)
        else:
            self.filters = parse_filter_list(self._filter_input)

        self.all_lightcurves, self.detrending_index_array = read_input_file(data_files)

    def run_retrieval(self, ld_fit_method='independent', fitting_mode='auto',
                      max_parameters=25, maxiter=None, maxcall=None,
                      sample='auto', nlive=300, dlogz=None, plot_final=True,
                      plot_partial=True,
                      results_output_folder='./output_parameters',
                      final_lightcurve_folder='./fitted_lightcurves',
                      plot_folder='./plots'):
        '''
        Runs dynesty on the data. Different modes exist and can be specified
        using the kwargs.

        Parameters
        ----------
        ld_fit_method : {`'coupled'`, `'single'`, `'independent'`}, optional
            Determines the mode of fitting of limb darkening parameters. The
            available modes are:
                - `'coupled'` : all limb darkening parameters are fitted
                  independently, but are coupled to a wavelength dependent
                  model based on the host parameters through `ldkt`
                - `'single'` : LD parameters are still tied to a model, but
                  only the first filter is actively fitted. The remaining
                  filters are estimated based off the ratios given by ldtk for
                  a host with the given parameters. This mode is useful for a
                  large number of filters, as `'coupled'` or `'independent'`
                  fitting will lead to much higher computation times.
                - `'independent'` : Each LD coefficient is fitted separately for
                  each filter, with no coupling to the ldtk models.
            Default is `'independent'`
        fitting_mode : {'auto', 'all', 'folded'}, optional
            Determines if the fitting algorithm is limited by max_parameters.
            If the number of parameters to be fitted exceeds max_parameters,
            then the retrieval will split into fitting each filter
            independently, phase-folding the detrended light curves to
            produce a single light curve for each filter and then fitting
            these phase-folded curves simultaneously. If fitting_mode is
            `'auto'`, then the mode used will be determined automatically.
            If fitting_mode is `'all'`, then all light curves will be
            attempted to be fitted simultaneously, regardless of the
            value of max_parameters. If fitting_mode is `'folded'`, then
            the folding approach will be used. Default is `'auto'`
        max_parameters : int, optional
            The maximum number of parameters to use in a single retrieval.
            Default is 25.
        maxiter : int or `None`, optional
            The maximum number of iterations to run. If `None`, will
            continue until stopping criterion is reached. Default is `None`.
        maxcall : int or `None`, optional
            The maximum number of likelihood calls in retrieval. If None, will
            continue until stopping criterion is reached. Default is `None`.
        sample : str, optional
            Method used to sample uniformly within the likelihood constraint,
            conditioned on the provided bounds. Unique methods available are:
            uniform sampling within the bounds('unif'), random walks with fixed
            proposals ('rwalk'), random walks with variable (“staggering”)
            proposals ('rstagger'), multivariate slice sampling along preferred
            orientations ('slice'), “random” slice sampling along all
            orientations ('rslice'), “Hamiltonian” slices along random
            trajectories ('hslice'), and any callable function which follows
            the pattern of the sample methods defined in dynesty.sampling.
            'auto' selects the sampling method based on the dimensionality of
            the problem (from ndim). When ndim < 10, this defaults to 'unif'.
            When 10 <= ndim <= 20, this defaults to 'rwalk'. When ndim > 20,
            this defaults to 'hslice' if a gradient is provided and 'slice'
            otherwise. 'rstagger' and 'rslice' are provided as alternatives for
            'rwalk' and 'slice', respectively. Default is 'auto'.
        nlive : int, optional
            The number of live points to use in the nested sampling retrieval.
            Default is 300.
        dlogz : float, optional
            Retrieval iteration will stop when the estimated contribution of
            the remaining prior volume to the total evidence falls below this
            threshold. Explicitly, the stopping criterion is
            `ln(z + z_est) - ln(z) < dlogz`, where z is the current evidence
            from all saved samples and z_est is the estimated contribution from
            the remaining volume. The default is `1e-3 * (nlive - 1) + 0.01`.
        '''

        if fitting_mode.lower() == 'auto':
            # Generate the basic PriorInfo, needed to calculate how many
            # parameters would be fitted
            complete_prior = self._generate_priorinfo('full', ld_fit_method)
            n_params_for_complete = len(complete_prior.fitting_params)

            if n_params_for_complete > max_parameters:
                fitting_mode = 'folded'
                print("Auto mode detect has set 'folded' mode")
            else:
                print("Auto mode detect has set 'all' mode")
                fitting_mode = 'all'

        if fitting_mode.lower() == 'all':
            # We are fitting everything simultaneously
            priors = self._generate_priorinfo('full', ld_fit_method)

            results, ndof = self._run_dynesty(self.all_lightcurves, priors,
                                              maxiter, maxcall, sample, nlive,
                                              dlogz)

            # Print results to terminal
            io.print_results(results, priors, ndof)

            # Save output
            try:
                save_fname = os.path.join(results_output_folder, 'output.csv')
                io.save_results(results, priorinfo, save_fname)
                print('Best fit parameters saved to {}'.format(os.path.abspath(save_fname)))
            except Exception as e:
                print('The following exception was raised whilst saving parameter results:')
                print(e)

            # Save final light curves
            try:
                output_folder = os.path.join(final_lightcurve_folder, 'all-mode_lightcurves')
                io.save_final_light_curves(self.all_lightcurves, priors,
                                           results, output_folder)
                print('Fitted light curves saved to {}'.format(os.path.abspath(output_folder)))
            except Exception as e:
                print('The following exception was raised whilst saving final light curves:')
                print(e)

            # Plot the final curves!
            if plot_final:
                try:
                    save_folder = os.path.join(plot_folder, 'all-mode_plots')
                    plot_individual_lightcurves(self.all_lightcurves, priorinfo,
                                                results, folder_path=save_folder,
                                                color=plot_color, fnames=None)
                    print('Plots saved to {}'.format(os.path.abspath(plot_folder)))
                except Exception as e:
                    # TODO: Try plotting from files rather than results objects
                    print('The following exception was raised whilst plotting final light curves:')
                    print(e)

            return results

        elif fitting_mode.lower() == 'folded':
            raise NotImplementedError

        else:
            raise ValueError('Unrecognised fitting mode {}'.format(fitting_mode))

    def _generate_priorinfo(self, mode, ld_fit_method, filter_index=None,
                           detrending=True, normalise=True):
        '''
        Generates a prior info for a particular run type:
        Modes available are
            'full' - we are doing no splitting
            'single filter' - we are only using one filter
            'folded' - we are using multiple filters with folded light curves
        '''

        #######################################################################
        #####                           FULL MODE                         #####
        #######################################################################
        if mode.lower() == 'full':
            # We are fitting everything in one. We don't need to mess about
            # with filters or epochs
            if type(self._prior_input) == str:
                # read in priors from a file
                priors = read_priors_file(self._prior_input, self.n_telescopes,
                                          self.n_filters, self.n_epochs,
                                          self.limb_darkening_model)
            else:
                # Reading in from a list
                priors = parse_priors_list(self._prior_input, self.n_telescopes,
                                           self.n_filters, self.n_epochs,
                                           self.limb_darkening_model)

            # Set up limb darkening
            if ld_fit_method == 'independent':
                priors.fit_limb_darkening(ld_fit_method)
            elif ld_fit_method in ['coupled', 'single']:
                if self._filter_input is None:
                    raise ValueError('filter_info must be provided for coupled and single ld_fit_methods')
                if self.host_T is None or self.host_z is None or self.host_logg is None:
                    raise ValueError('Filter info was provided but I am missing information on the host!')

                priors.fit_limb_darkening(ld_fit_method, self.host_T,
                                          self.host_logg, self.host_z,
                                          self.filters,
                                          cache_path=self.ldtk_cache)
            # Set up detrending
            if detrending:
                priors.fit_detrending(self.all_lightcurves,
                                      self.detrending_info,
                                      self.detrending_index_array)

            # Set up normalisation
            if normalise:
                priors.fit_normalisation(self.all_lightcurves)

            return priors

        #######################################################################
        #####                       SINGLE FILTER                         #####
        #######################################################################
        elif mode.lower() == 'single filter':
            raise NotImplementedError


        #######################################################################
        #####                           FOLDED                            #####
        #######################################################################
        elif mode.lower() == 'folded':
            raise NotImplementedError


        else:
            raise ValueError('Unrecognised mode {}'.format(mode))


    def _run_dynesty(self, lightcurves, priors, maxiter=None, maxcall=None,
                     sample='auto', nlive=300, dlogz=None):
        '''
        Invokes a run of dynesty and returns the results object. Note that this
        does not save data, plot or print.

        Returns
        -------
        results

        ndof
        '''
        # Get number of dimensions and degrees of freedom
        n_dims = len(priors.fitting_params)

        n_dof = 0
        for i in np.ndindex(lightcurves.shape):
            if lightcurves[i] is not None:
                n_dof += len(lightcurves[i].times)

        # Make a LikelihoodCalculator
        likelihood_calc = LikelihoodCalculator(lightcurves, priors)

        #######################################################################
        #######################################################################
        # Now we define the prior transform and ln likelihood function for
        # dynesty to use
        def prior_transform(cube):
            return priors._convert_unit_cube(cube)

        def lnlike(cube):
            params = priors._interpret_param_array(cube)

            # Get the limb darkening details and coefficient values
            limb_dark = priors.limb_dark
            u = [params[key] for key in priors.limb_dark_coeffs]

            if priors.detrend:
                # We need to combine the detrending coeff arrays into one
                # Each entry should be a list containing all the detrending
                # coefficients to trial.
                d = np.full(lightcurves.shape, None, object)

                for i in np.ndindex(d.shape):
                    for coeff in priors.detrending_coeffs:
                        if params[coeff][i] is not None:
                            if d[i] is None:
                                d[i] = [params[coeff][i]]
                            else:
                                d[i].append(params[coeff][i])

            else:
                # Don't detrend
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
                                                            d)
            if priors.fit_ld and not priors.ld_fit_method == 'independent':
                return ln_likelihood + priors.ld_handler.ldtk_lnlike(np.array(u).T, limb_dark)
            else:
                return ln_likelihood
        #######################################################################
        #######################################################################

        # Now we can set up and run the sampler!
        sampler = NestedSampler(lnlike, prior_transform, n_dims, bound='multi',
                                sample=sample, update_interval=float(n_dims),
                                nlive=nlive)
        sampler.run_nested(maxiter=maxiter, maxcall=maxcall, dlogz=dlogz)

        # Pull out the results and calculate a few additional bits of info
        results = sampler.results

        # Normalise weights
        results.weights = get_normalised_weights(results)

        # Calculate covariance matrix and use to get uncertainties
        cov = get_covariance_matrix(results)
        diagonal = np.diag(cov)
        uncertainties = np.sqrt(diagonal)

        results.cov = cov
        results.uncertainties = uncertainties

        return results, n_dof
