'''
Retriever new
- updated to use ParamArray approach

'''

import numpy as np
from dynesty import NestedSampler
from copy import deepcopy
import os, csv
import traceback

from transitfit import io


from .plotting import plot_individual_lightcurves, quick_plot
from .detrending_funcs import NthOrderDetrendingFunction
from .detrender import DetrendingFunction
from .io import read_input_file, read_priors_file, parse_priors_list, read_filter_info, parse_filter_list
from ._likelihood import LikelihoodCalculator
from ._utils import get_normalised_weights, get_covariance_matrix, validate_lightcurve_array_format, weighted_avg_and_std

# Parameters and if they are global, filter-specific or lightcurve-specific
global_params = ['P', 'ecc', 'a', 'inc', 'w']

filter_dependent_params = ['rp', 'q0', 'q1', 'q2', 'q3', 'u0', 'u1', 'u2', 'u3']

lightcurve_dependent_params = ['norm','d0','d1','d2','d3','d4','d5','d6','d7','d8']

class Retriever:
    def __init__(self, data_files, priors, n_telescopes, n_filters, n_epochs,
                 filter_info=None, detrending_list=[['nth order', 1]],
                 limb_darkening_model='quadratic', host_T=None, host_logg=None,
                 host_z=None, host_r=None, ldtk_cache=None, data_skiprows=0,
                 n_ld_samples=20000, do_ld_mc=False, fit_ttv=False):
        '''

        '''
        ###################
        # Save input data #
        ###################
        self._data_input = data_files
        self._prior_input = priors
        self._filter_input = filter_info

        self.n_telescopes = n_telescopes
        self.n_filters = n_filters
        self.n_epochs = n_epochs

        self.fit_ttv = fit_ttv

        self.detrending_info = detrending_list

        # Host info
        self.host_T = host_T
        self.host_logg = host_logg
        self.host_z = host_z
        self.host_r = host_r

        # Limb darkening things
        self.limb_darkening_model = limb_darkening_model
        self.n_ld_samples = n_ld_samples
        self.do_ld_mc = do_ld_mc
        self.ldtk_cache = ldtk_cache

        ###########################################################################
        # Now read in things from files to get Filters, full priors, full lc data #
        ###########################################################################
        # Read in the filters
        if self._filter_input is None:
            self.filters = None
        elif type(self._filter_input) == str:
            self.filters = read_filter_info(self._filter_input)
        else:
            self.filters = parse_filter_list(self._filter_input)

        # Load in the full LightCurve data and detrending index array
        self.all_lightcurves, self.detrending_index_array = read_input_file(data_files, data_skiprows)

        # Get the FULL prior.
        # Assume independent ld method - it doesn't actually matter
        # since this never gets used.
        self._full_prior, _ = self._get_priors_and_curves(self.all_lightcurves, 'independent')

        # From the full prior, get a bunch of info on numbers of parameters
        # Get the number of parameters in the limb darkening
        self.n_ld_params = len(self._full_prior.limb_dark_coeffs)
        self.ld_coeffs = self._full_prior.limb_dark_coeffs

        # Get the number of globally fitted parameters
        # Work out the possible global parameters based on fit_ttv
        global_params = ['P', 'ecc', 'a', 'inc', 'w']
        if not self.fit_ttv:
            global_params.append('t0')
        self.n_global_params = 0
        for param in global_params:
            if param in np.array(self._full_prior.fitting_params)[:,0]:
                self.n_global_params += 1

        # Calculate the detrending models - used in calculating the number of
        # parameters used in a fitting batch
        self.detrending_models = []
        for mode in self.detrending_info:
            if mode[0] == 'nth order':
                function = NthOrderDetrendingFunction(mode[1])
                self.detrending_models.append(DetrendingFunction(function))
            elif mode[0] == 'custom':
                function = mode[1]
                self.detrending_models.append(DetrendingFunction(function))
            elif mode[0] == 'off':
                self.detrending_models.append(None)

    ##########################################################
    #               RETRIEVAL RUNNING METHODS                #
    ##########################################################
    def _run_dynesty(self, lightcurves, priors, maxiter=None, maxcall=None,
                     sample='auto', nlive=300, dlogz=None):
        '''
        Runs dynesty on the given lightcurves with the given priors. Returns
        the result.

        Returns
        -------
        results
        ndof
        '''
        # Get number of dimensions and degrees of freedom
        n_dims = len(priors.fitting_params)

        # Calculate the number of degrees of freedom - how many data points do we have?
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

            ln_likelihood = likelihood_calc.find_likelihood(params)

            if priors.fit_ld and not priors.ld_fit_method == 'independent':
                # Pull out the q values and convert them
                u = []
                for fi in range(priors.n_filters):
                    q = [params[qX][0,fi,0] for qX in priors.limb_dark_coeffs]
                    u.append(priors.ld_handler.convert_qtou(*q))

                return ln_likelihood + priors.ld_handler.ldtk_lnlike(u, limb_dark)
            else:
                return ln_likelihood
        #######################################################################
        #######################################################################

        # Now we can set up and run the sampler!
        sampler = NestedSampler(lnlike, prior_transform, n_dims, bound='multi',
                                sample=sample, #update_interval=float(n_dims),
                                nlive=nlive)

        try:
            print('Retrieving {} parameters:'.format(n_dims))
            sampler.run_nested(maxiter=maxiter, maxcall=maxcall, dlogz=dlogz)
        except:
            raise


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

        # Save the best fit results for easy access
        results.best = results.samples[np.argmax(results.logl)]

        return results, n_dof

    def _run_full_retrieval(self, ld_fit_method, detrend, normalise, maxiter,
                            maxcall, sample, nlive, dlogz,
                            output_folder='./output_parameters',
                            summary_file='summary_output.csv',
                            full_output_file='full_output.csv',
                            lightcurve_folder='./fitted_lightcurves',
                            plot=True, plot_folder='./plots',
                            marker_color='dimgrey', line_color='black'):
        '''
        Runs full retrieval with no folding/batching etc. Just a straight
        forward dynesty run.
        '''
        priors, lightcurves = self._get_priors_and_curves(ld_fit_method,
                                                          detrend=detrend,
                                                          normalise=normalise)

        results, ndof = self._run_dynesty(lightcurves, priors,
                                          maxiter, maxcall, sample, nlive,
                                          dlogz)

        # Print results to terminal
        try:
            io.print_results(results, priors, ndof)
        except Exception as e:
            print(e)

        # Save outputs parameters, plots, lightcurves!
        self._save_results([results], [priors], [lightcurves], output_folder,
                          summary_file, full_output_file, lightcurve_folder,
                          plot, plot_folder, marker_color, line_color)

        return results

    def _run_batched_retrieval(self, lightcurves, batches, ld_fit_method, detrend,
                               normalise, maxiter, maxcall, sample, nlive,
                               dlogz, full_return=False,
                               folded=False,
                               folded_P=None, folded_t0=None,
                               output_folder='./output_parameters',
                               summary_file='summary_output.csv',
                               full_output_file='full_output.csv',
                               lightcurve_folder='./fitted_lightcurves',
                               plot=True, plot_folder='./plots',
                               marker_color='dimgrey', line_color='black'):
        '''
        Runs a retrieval using the given batches

        lightcurves : array_like, shape (n_telescopes, n_filters, n_epochs)
            The full lightcurves array to be retrieved.
        full_return : bool, optional
            If True will return all_results, all_priors, all_lightcurves.
            If False, will just return all_results. Default is False
        '''
        all_results = []
        all_priors = []
        all_lightcurves = []

        for bi, batch in enumerate(batches):
            print('Batch {} of {}'.format(bi+1, len(batches)))
            # Now we want to get the lightcurves and priors for each batch
            batch_prior, batch_lightcurves = self._get_priors_and_curves(lightcurves, ld_fit_method, batch, detrend, normalise, folded, folded_P, folded_t0)

            print(batch_prior)
            # Run the retrieval!
            results, ndof = self._run_dynesty(batch_lightcurves, batch_prior, maxiter,
                                              maxcall, sample, nlive,
                                              dlogz)

            all_results.append(results)
            all_priors.append(batch_prior)
            all_lightcurves.append(batch_lightcurves)

        # Make outputs etc
        self._save_results(all_results, all_priors, all_lightcurves,
                          output_folder, summary_file, full_output_file,
                          lightcurve_folder, plot, plot_folder, marker_color,
                          line_color)

        if full_return:
            return all_results, all_priors, all_lightcurves

        return all_results

    def _run_folded_retrieval(self, ld_fit_method, detrend, normalise, maxiter,
                              maxcall, sample, nlive, dlogz,
                              output_folder='./output_parameters',
                              summary_file='summary_output.csv',
                              full_output_file='full_output.csv',
                              lightcurve_folder='./fitted_lightcurves',
                              plot=True, plot_folder='./plots',
                              marker_color='dimgrey', line_color='black',
                              max_parameters=25, overlap=2):
        '''
        For each filter, runs retrieval, then produces a phase-folded
        lightcurve. Then runs retrieval across wavelengths on the folded
        curves.
        '''

        prefolded_batches = self._get_folding_batches(max_parameters, detrend, normalise, overlap)

        print('Prefolded_batches:', prefolded_batches)

        # Blank lists to fill with results etc
        #results_list = [[] for i in range(self.n_filters)]
        #priors_list = [[] for i in range(self.n_filters)]
        #lightcurve_list = [[] for i in range(self.n_filters)]
        results_list = []
        priors_list = []
        lightcurve_list = []

        print('Running pre-folding retrievals:')
        # Now run the batches for each filter
        for fi, filter_batches in enumerate(prefolded_batches):
            print('Filter {} of {}'.format(fi + 1, self.n_filters))

            # Make a bunch of paths etc for saving the partial results
            filter_output_folder = os.path.join(output_folder, 'filter_{}_parameters'.format(fi))
            filter_summary = 'filter_{}_summary.csv'.format(fi)
            filter_full_output = 'filter_{}_full_output.csv'.format(fi)
            filter_lightcurve_folder = os.path.join(lightcurve_folder, 'filter_{}_curves'.format(fi))
            filter_plots_folder = os.path.join(plot_folder, 'filter_{}_plots'.format(fi))

            results, priors, lightcurves = self._run_batched_retrieval(self.all_lightcurves,
                                            filter_batches,
                                            ld_fit_method,
                                            detrend, normalise,
                                            maxiter, maxcall,
                                            sample, nlive, dlogz,
                                            full_return=True,
                                            output_folder=filter_output_folder,
                                            summary_file=filter_summary,
                                            full_output_file=filter_full_output,
                                            lightcurve_folder=filter_lightcurve_folder,
                                            plot=plot, plot_folder=filter_plots_folder,
                                            marker_color=marker_color, line_color=line_color)

            results_list.append(results)
            priors_list.append(priors)
            lightcurve_list.append(lightcurves)

        # Now fold the lightcurves
        print('Folding lightcurves...')
        folded_curves, folded_P, folded_t0 = self._fold_lightcurves(results_list, priors_list, lightcurve_list)

        print(folded_curves, folded_curves.shape)

        # Plot the folded lightcurves so we can check them
        for lci, lc in np.ndenumerate(folded_curves):
            print(lci, lc)
            quick_plot(lc, 'folded_curve_filter_{}.pdf'.format(lci[1]), os.path.join(plot_folder, 'folded_curves'), folded_t0[lci[1]])

        # Get the batches, and remember that now we are not detrending or
        # normalising since that was done in the first stage
        folded_batches = self._get_non_folding_batches(folded_curves, max_parameters, ld_fit_method,
                                     detrend=False, normalise=False, overlap=overlap)

        print('Running folded retrievals...')
        return self._run_batched_retrieval(folded_curves, folded_batches, ld_fit_method, False,
                        False, maxiter, maxcall, sample, nlive, dlogz,
                        False, True, folded_P, folded_t0, output_folder=output_folder,
                        summary_file=summary_file, full_output_file=full_output_file,
                        lightcurve_folder=lightcurve_folder, plot=plot, plot_folder=plot_folder,
                        marker_color=marker_color, line_color=line_color)

    def run_retrieval(self, ld_fit_method='independent', fitting_mode='auto',
                      max_parameters=25, maxiter=None, maxcall=None,
                      sample='auto', nlive=300, dlogz=None, plot=True,
                      output_folder='./output_parameters',
                      lightcurve_folder='./fitted_lightcurves',
                      summary_file='summary_output.csv',
                      full_output_file='full_output.csv',
                      plot_folder='./plots', marker_color='dimgray',
                      line_color='black',
                      normalise=True, detrend=True, overlap=2):
        '''
        Runs dynesty on the data. Different modes exist and can be specified
        using the kwargs.

        Parameters
        ----------
        ld_fit_method : {`'coupled'`, `'single'`, `'independent'`, `'off'`}, optional
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
        fitting_mode : {'auto', 'all', 'folded', 'batched'}, optional
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
        # Auto mode detect
        if fitting_mode.lower() == 'auto':
            # Generate the basic PriorInfo, needed to calculate how many
            # parameters would be fitted
            n_params_for_complete = self._calculate_n_params(self.all_lightcurves,
                                                             None, ld_fit_method,
                                                             normalise,
                                                             detrend)

            if n_params_for_complete > max_parameters and not self.n_total_lightcurves == 1:
                # We have more than one lightcurve to fit - we can batch
                # If one filter has >= 3 epochs, we can do folded mode
                n_epochs_in_filter = np.sum(self.all_lightcurves!=None, axis=(2,0))

                if np.any(n_epochs_in_filter >= 3):
                    print("Auto mode detect has set 'folded' mode")
                    fitting_mode = 'folded'
                else:
                    print("Auto mode detect has set 'batched' mode")
                    fitting_mode = 'batched'
            else:
                print("Auto mode detect has set 'all' mode")
                fitting_mode = 'all'

        if fitting_mode.lower() == 'all':
            # In this mode, we are just running everything through a single
            # dynesty retrieval, not batching or anything.
            return self._run_full_retrieval(ld_fit_method, detrend, normalise,
                    maxiter, maxcall, sample, nlive, dlogz, output_folder,
                    summary_file, full_output_file, lightcurve_folder, plot,
                    plot_folder, marker_color, line_color)

        if fitting_mode.lower() == 'batched':
            # In this mode, we are generating batches which contain all
            # lightcurves for a filter. Batches may contain more than one
            # filter, but filters will not be split across multiple batches

            # Calculate the batches
            batches = self._get_non_folding_batches(self.all_lightcurves, max_parameters, detrend, normalise, overlap)

            return self._run_batched_retrieval(self.all_lightcurves, batches, ld_fit_method,detrend,
                    normalise, maxiter, maxcall, sample, nlive, dlogz, False,
                    False, None, None, output_folder, summary_file,
                    full_output_file, lightcurve_folder, plot, plot_folder,
                    marker_color, line_color, max_parameters, overlap)

        if fitting_mode.lower() == 'folded':
            # In this mode, we are running multi-epoch retrieval on each
            # filter separately, and then producing phase folded lightcurves
            # to run a multi-filter retrieval on.
            return self._run_folded_retrieval(ld_fit_method, detrend, normalise,
                    maxiter, maxcall, sample, nlive, dlogz, output_folder,
                    summary_file, full_output_file, lightcurve_folder, plot,
                    plot_folder, marker_color, line_color, max_parameters, overlap)

    ##########################################################
    #            PRIOR MANIPULATION                          #
    ##########################################################
    def _get_priors_and_curves(self, lightcurves, ld_fit_method, indices=None,
                               detrend=True, normalise=True, folded=False,
                               folded_P=None, folded_t0=None):
        '''
        Generates a prior info for a particular run:

        Parameters
        ----------
        ld_fit_method : {'independent', 'single', 'coupled', 'off'}
            The mode to fit limb darkening coefficients with.
        indices : tuple or None
            If None, will fit all light curves. Otherwise, supply relevant
            indices of lightcurves to fit as a tuple:
            (telescope_indices, filter_indices, epoch_indices)
        detrend : bool, optional
            If True, will initialise detrending fitting. Default is True.
        normalise : bool, optional
            If True, will initialise normalisation fitting. Default is True.
        folded : bool, optional
            Set to True if using folded light curves (functionally only one
            epoch). Also turns off detrending and normalisation fitting.
            Default is False.
        folded_P : float, optional
            Required if folded is True. This is the period that the light
            curves are folded to
        folded_t0 : float, optional
            Required if folded is True. This is the t0 that the light curves
            are folded to. In the case where ttv mode is used (each epoch
            has been fitted to a different t0), then this should be the t0 of
            the first epoch (the one which everything else is folded back to)

        Returns
        -------
        priors : PriorInfo
            The fully initialised PriorInfo object
        lightcurves :
            The LightCurves in the correct format, with detrending and
            normalisation initialised

        '''
        # Sort out indices and number of filters etc here.
        indices = self._format_indices(indices)

        unique_indices = self._get_unique_indices(indices)

        # Indices of filters being used
        filter_indices = unique_indices[1]

        if folded:
            if filter_indices is None:
                raise ValueError('filter_indices must be provided for folded PriorInfo')

            if lightcurves is None:
                raise ValueError('lightcurves must be provided for folded PriorInfo')

            lightcurve_subset = self._get_lightcurve_subset(lightcurves, indices)

            n_telescopes = 1
            n_filters = len(filter_indices)
            n_epochs = 1
            detrend = False
            normalise = False

        else:
            # get the lightcurve and detrending index array into the right shapes
            # and extract the relevant info.
            lightcurve_subset = self._get_lightcurve_subset(self.all_lightcurves, indices)
            detrending_indices = self._get_detrending_subset(indices)

            # Get unique indices and work out number of filters etc
            n_telescopes = len(unique_indices[0])
            n_filters = len(unique_indices[1])
            n_epochs = len(unique_indices[2])


        # Set up the basic PriorInfo
        if type(self._prior_input) == str:
            # read in priors from a file
            priors = read_priors_file(self._prior_input,
                                      n_telescopes,
                                      n_filters,
                                      n_epochs,
                                      self.limb_darkening_model,
                                      filter_indices,
                                      folded, folded_P, folded_t0, self.host_r,
                                      self.fit_ttv, lightcurve_subset)
        else:
            # Reading in from a list
            priors = parse_priors_list(self._prior_input,
                                       n_telescopes,
                                       n_filters,
                                       n_epochs,
                                       self.limb_darkening_model,
                                       filter_indices,
                                       folded, folded_P, folded_t0,
                                       self.host_r, self.fit_ttv, lightcurve_subset)

        # Set up limb darkening
        if not ld_fit_method.lower() == 'off':
            if ld_fit_method.lower() == 'independent':
                priors.fit_limb_darkening(ld_fit_method)
            elif ld_fit_method.lower() in ['coupled', 'single']:
                if self._filter_input is None:
                    raise ValueError('filter_info must be provided for coupled and single ld_fit_methods')
                if self.host_T is None or self.host_z is None or self.host_logg is None:
                    raise ValueError('Filter info was provided but I am missing information on the host!')
                priors.fit_limb_darkening(ld_fit_method, self.host_T,
                                          self.host_logg, self.host_z,
                                          self.filters[filter_indices],
                                          self.n_ld_samples, self.do_ld_mc,
                                          self.ldtk_cache,)
            else:
                raise ValueError("Unrecognised ld_fit_method '{}'".format(ld_fit_method))

        if detrend:
            priors.fit_detrending(lightcurve_subset,
                                  self.detrending_info, detrending_indices)

        # Set up normalisation
        if normalise:
            priors.fit_normalisation(lightcurve_subset)

        return priors, lightcurve_subset

    ##########################################################
    #          LIGHTCURVE MANIPULATION                       #
    ##########################################################
    def _fold_lightcurves(self, results, priors, lightcurves):
        '''
        Produces a set of folded lightcurves which can then be fitted across
        filters

        Parameters
        ----------
        results : array_like, shape (n_filters, )
            Each entry should be a list of results objects for each batch
        priors : array_like, shape (n_filters, )
            Each entry should be a list of PriorInfo objects for each batch
        lightcurves : array_like, shape (n_filters, )
            The lightcurves. Each entry should be a list of LightCurve arrays
            used for the batches within each filter

        Returns
        -------
        folded_lightcurves : np.array, shape (1, n_filters, 1)
            All the lightcurves, with each filter folded onto one epoch.
        folded_P : float
            The period that the lightcurves are folded with
        folded_t0 : float
            The t0 that all lightcurves are centred on.
        '''
        # For each filter, go through the results and extract the best fit
        # values and uncertainties. For global values, we take the weighted
        # average and then use the detrending, normalisation, P and t0 values
        # to produce a single LightCurve comprised by folding all the
        # detrended and normalised curves within a filter

        n_filters = self.n_filters
        fit_ttv = self.fit_ttv

        # Blank arrays to store the global parameter retrievals
        retrieved_P = []
        retrieved_P_err = []

        # We need t0 for each epoch
        retrieved_t0 = [[] for f in range(self.n_epochs)]
        retrieved_t0_err = [[] for f in range(self.n_epochs)]

        ###############################################################
        ###                  GET P AND t0 VALUES                    ###
        ###############################################################
        for fi , filter_results in enumerate(results):
            for ri, result in enumerate(filter_results):
                prior = priors[fi][ri]

                # Get the results and errors dictionaries
                results_dict, errors_dict = prior._interpret_final_results(result)

                for i in np.ndindex(lightcurves[fi][ri].shape):

                    tidx = lightcurves[i].telescope_idx
                    fidx = lightcurves[i].filter_idx
                    eidx = lightcurves[i].epoch_idx

                    retrieved_P.append(results_dict['P'][i])
                    retrieved_P_err.append(errors_dict['P'][i])

                    retrieved_t0[eidx].append(results_dict['t0'][i])
                    retrieved_t0_err[eidx].append(errors_dict['t0'][i])

        single_val = not self.fit_ttv

        print(retrieved_t0)
        print(retrieved_t0_err)

        # Find the weighted average to get the best fit values of P and t0
        best_P, P_err = weighted_avg_and_std(retrieved_P, retrieved_P_err, single_val=True)
        best_t0, t0_err = weighted_avg_and_std(retrieved_t0, retrieved_t0_err, single_val=single_val)

        print(best_t0, t0_err)

        if not self.fit_ttv:
            # Put the t0 values and errors into an array just for simplicity
            # in dealing with the two modes
            best_t0 = best_t0 * np.ones(self.n_epochs)
            t0_err = t0_err * np.ones(self.n_epochs)
        print(best_t0, t0_err)

        print('P = {} ± {}'.format(round(best_P, 8),  round(P_err, 8)))
        if self.fit_ttv:
            for i, t0i in enumerate(best_t0):
                print('t0 = {} ± {} (epoch {})'.format(round(t01, 3),  round(t0_err[i], 3), i))
        else:
            print('t0 = {} ± {}'.format(best_t0.round(3)[0],  t0_err.round(3)[0]))

        ###############################################################
        ###            NORMALISATION/DETRENDING/FOLDING             ###
        ###############################################################
        # Now we do the detrending stuff and fold the lightcurves
        # Remember that each batch will only contain one filter

        final_batched_lightcurves = [[] for i in range(self.n_filters)]

        for fi, filter_results in enumerate(results):
            # Loop through each filter
            for ri, result in enumerate(filter_results):
                # Loop through each batch within a filter

                # Get the batch-relevant prior, lightcurves and results dicts
                prior = priors[fi][ri]
                lcs = lightcurves[fi][ri]

                results_dict, errors_dict = prior._interpret_final_results(result)

                # Loop through lightcurves and detrend/normalise
                batch_detrended_curves = []
                for i, lc in np.ndenumerate(lcs):
                    # Get some sub indices.
                    tidx, fidx, eidx = i
                    if prior.detrend:
                        method_idx = lc.detrending_method_idx
                        # Get the detrending coeffs for this lc
                        detrending_coeffs = prior.detrending_coeffs[method_idx]

                        best_d = [results_dict[d][i] for d in detrending_coeffs]
                    else:
                        best_d = None

                    norm = results_dict['norm'][i]

                    # Make the detrended light curve and fold to the final t0
                    detrended_curve = lc.create_detrended_LightCurve(best_d, norm)

                    # Fold the curve using the best t0 for the epoch and P
                    # We are folding each curve to be centred on best_t0[0]
                    folded_curve = detrended_curve.fold(best_t0[eidx], best_P, best_t0[0])

                    batch_detrended_curves.append(folded_curve)

            final_batched_lightcurves[fi].append(batch_detrended_curves)

        # Now we go through detrended and folded lightcurve, and combine them
        # into one lightcurve per filter
        # TODO - complete this function
        final_lightcurves = []
        for fi, filter_curves in enumerate(final_batched_lightcurves):
            # Go through each filter and combine the curves!
            if len(filter_curves) == 1:
                # No need to combine
                final_lightcurves.append(filter_curves[0])
            else:
                # Need to loop and combine
                final_lightcurves.append(filter_curves[0].combine(filter_curves[1:]))

        return np.array(final_lightcurves).reshape(1, self.n_filters, 1), best_P, best_t0

    def _get_lightcurve_subset(self, lightcurves, indices):
        '''
        Pulls out the subset of lightcurves given by the indices.
        '''
        # Get the unique indices
        unique_indices = self._get_unique_indices(indices)
        # Make some empty arrays for us to populate
        lightcurves_subset = np.full(tuple(len(idx) for idx in unique_indices), None)

        # Go through each index being used and put in the relevant info
        # deepcopy is used here to ensure we don't end up with clashing
        # attributes (e.g. from detrending twice)
        for index in np.array(indices).T:
            subset_index = self._full_to_subset_index(indices, index)
            lightcurves_subset[subset_index] = deepcopy(lightcurves[tuple(index)])

        return lightcurves_subset

    def _get_detrending_subset(self, indices):
        '''
        Pulls out the detrending indices for the given
        telescope, filter and epoch indices (given as tuple), assuming that we
        are working on all input lightcurves, not a folded version (though that
        shouldn't be using this function)
        '''
        # Get the unique indices
        unique_indices = self._get_unique_indices(indices)

        detrending_indices = np.full(tuple(len(idx) for idx in unique_indices), None)

        # Go through each index being used and put in the relevant info
        # deepcopy is used here to ensure we don't end up with clashing
        # attributes (e.g. from detrending twice)
        for index in np.array(indices).T:
            subset_index = self._full_to_subset_index(indices, index)
            detrending_indices[subset_index] = deepcopy(self.detrending_index_array[tuple(index)])

        return detrending_indices

    ##########################################################
    #        FUNCTIONS FOR BATCHING ETC                      #
    ##########################################################
    def _get_folding_batches(self, max_parameters, ld_fit_method,
                             detrend, normalise, overlap=2):
        '''
        Splits all_lightcurves into single-filter, multi-epoch batches
        to be fitted, which will allow us to produce folded lightcurves. This
        includes the option to have batches overlapping so that they share some
        info.

        Parameters
        ----------
        max_parameters : int
            The maximum number of parameters to have in a single batch
        ld_fit_method : str
            The limb darkening fit method
        detrend : bool
            If True, detrending will be used
        normalise : bool
            If true, normalisation will be used
        overlap : int, optional
            The number of epochs to overlap in each batch. This will be adhered
            to where possible. Default is 2.

        Returns
        -------
        batches : array_like, shape (n_filters,)
            Each entry in the array is a list of batches for the given filter.
        '''
        # All of the batches. Each entry will be the batches for a filter
        all_batches = []

        # Loop through each filter
        for fi in range(self.n_filters):
            # All the batches for this filter
            filter_batches = []

            # Find the indices of the lightcurves within the filter
            indices = np.where(self.all_lightcurves[:,fi,:] != None)
            n_curves = len(indices[0])

            # Put the fi index back into indices tuple
            indices = (indices[0], np.full(n_curves, fi), indices[1])

            # Loop through each lightcurve to get the batches!
            # What index are we starting the loop from? This is used to get the
            # overlapping batches
            start_idx = 0

            # Flag to check if we have got batches containing the full range
            # of epoch/filter
            done = False

            while not done:
                # A single batch - telescope, filter, epoch indices
                single_batch = ([], [], [])
                for i in range(start_idx, n_curves):
                    idx = (indices[0][i], indices[1][i], indices[2][i])
                    # Make the test batch by appending this lightcurve to the
                    # current single_batch
                    test_batch = tuple((single_batch[j] + [idx[j]] for j in range(3)))
                    # Get the number of parameters for the test_batch
                    n_params = self._calculate_n_params(self.all_lightcurves, test_batch, ld_fit_method,
                                                        normalise, detrend)

                    if n_params <= max_parameters:
                        # Add the lightcurve to the single batch and carry on
                        single_batch = deepcopy(test_batch)

                        if i == (n_curves - 1):
                            # We have completed the loop - we have all the
                            # lightcurves in at least one batch
                            done = True

                            # Save this final batch
                            filter_batches.append(single_batch)

                    else:
                        # Is this the only lightcurve in the batch?
                        # If so, we just want to add this curve and move on
                        if len(test_batch[0]) == 1:
                            single_batch = deepcopy(test_batch)
                            start_idx = i + 1

                        else:
                            # We will exlude the last added lightcurve and work
                            # only with the current confirmed batch

                            # Do we have enough to ensure an overlap?
                            if len(single_batch[0]) > overlap:
                                start_idx = i - overlap

                            else:
                                print('Unable to ensure overlap of {} between batch {} and {} for filter {}'.format(overlap, len(filter_batches), len(filter_batches) + 1, fi))

                                if len(single_batch[0]) > 1:
                                    # We can at least try for an overlap of 1
                                    start_idx = i - 1

                                else:
                                    # We just can't overlap
                                    start_idx = i

                        # Save the batch to start a new one
                        filter_batches.append(single_batch)

                        # Now check to see if we have done all the lightcurves
                        # for the filter
                        done = start_idx == n_curves
                        break

            all_batches.append(filter_batches)
        return all_batches

    def _get_non_folding_batches(self, lightcurves, max_parameters,
                                 ld_fit_method, detrend, normalise, overlap=2):
        '''
        Splits lightcurves into batches by filter, attempting to ensure that
        batches do not require more than max_parameters to fit. Each batch will
        contain every light curve in the filter, which trumps max_parameters.
        Where possible, the filter batches will overlap, allowing each batch
        to share information on some level.

        Parameters
        ----------
        lightcurves : array_like, shape (n_telescopes, n_filters, n_epochs)
            The LightCurves
        max_parameters : int
            The maximum number of parameters to have in a single batch
        ld_fit_method : str
            The limb darkening fit method
        detrend : bool
            If True, detrending will be used
        normalise : bool
            If true, normalisation will be used
        overlap : int, optional
            The number of epochs to overlap in each batch. This will be adhered
            to where possible. Default is 2.
        lightcurves : array_like, shape (n_telescopes, n_filters, n_epochs)
            The lightcurves to be batched.

        Returns
        -------
        batches : array_like, shape (n_batches)
            The final batches. Each entry is a tuple of
            (telescope indices, filter indices, epoch indices) using the
            indices of lightcurves. Each of these batches can then be passed to
            _get_priors_and_curves.
        '''
        # Get some info out of the lightcurves
        n_telescopes, n_filters, n_epochs = lightcurves.shape

        # The final batches
        all_batches = []

        # What filter are we starting the loop from? This is used to get the
        # overlapping batches
        start_idx = 0

        # Flag to check if each filter is in at least one batch
        done = False

        while not done:
            # A single batch - telescope, filter, epoch indices
            single_batch = ([], [], [])

            filters_in_batch = 0
            # loop through each filter
            for fi in range(start_idx, n_filters):

                # For each filter, we need to pull out all the light curves
                # and calculate how many parameters we would be fitting if
                # this filter were added to the batch.

                # First, get the indices of lightcurves in the filter:
                indices = np.where(lightcurves[:,fi,:] != None)
                n_curves = len(indices[0])

                # Put the fi index back into indices tuple, converting to list
                indices = (list(indices[0]), list(np.full(n_curves, fi)), list(indices[1]))

                # Make the test batch
                test_batch = tuple((single_batch[j] + indices[j] for j in range(3)))

                # Get the number of params for the test batch
                n_params = self._calculate_n_params(lightcurves, test_batch, ld_fit_method,
                                                    normalise, detrend)

                if n_params <= max_parameters:
                    # Add the filter to the batch
                    single_batch = deepcopy(test_batch)
                    filters_in_batch += 1
                    # Check to see if we have completed the loop and have all
                    # the filters in at least one batch
                    if fi == (self.n_filters - 1):
                        done = True
                        # save the final batch
                        all_batches.append(single_batch)

                else:
                    # Too many params - what to do?
                    # Is this the only filter in the batch? If so, add it and
                    # move on as there's too many parameters to put this
                    # filter in with anything else
                    if filters_in_batch == 0:
                        filters_in_batch += 1
                        single_batch = deepcopy(test_batch)

                        start_idx = fi + 1

                    else:
                        # Exclude the last added filter - don't update
                        # single_batch

                        # Do we have enough to ensure an overlap?
                        if filters_in_batch > overlap:
                            start_idx = fi - overlap

                        else:
                            print('Unable to ensure overlap of {} between batch {} and batch {}'.format(overlap, len(all_batches), len(all_batches) + 1))

                            # Can we at least try to get an overlap of 1?
                            if filters_in_batch > 1:
                                print('Attempting overlap of 1')
                                start_idx = fi - 1
                            else:
                                print('Overlap cannot be made')
                                # We just can't overlap. Ah well
                                start_idx = fi

                    # save the batch and start a new one
                    all_batches.append(single_batch)

                    # Now check to see if all the filters are included in
                    # at least one batch
                    done = start_idx == n_filters
                    break

        return all_batches

    ##########################################################
    #            OUTPUT FUNCTIONS                            #
    ##########################################################
    def _save_results(self, results, priors, lightcurves,
                      output_folder='./output_parameters',
                      summary_file='summary_output.csv',
                      full_output_file='full_output.csv',
                      lightcurve_folder='./fitted_lightcurves',
                      plot=True, plot_folder='./plots',
                      marker_color='dimgrey', line_color='black',
                      folded_P=None, folded_P_err=None, folded_t0=None,
                      folded_t0_err=None, fit_ld=True):
        '''
        Saves the parameters, as well as the detrended lightcurves and the
        best fit.

        Parameters
        ----------
        results : array_like, shape (n_batches, )
            The results for each of the runs
        priors : array_like, shape (n_batches, )
            The priors for each of the runs
        lightcurves : array_like, shape (n_batches, )
            Each entry should be the (n_telescopes, n_filters, n_epochs)
            lightcurve array for the batch
        output_folder : str, optional
            The folder to save output files to (not plots). Default is
            ./output_parameters
        summary_file : str, optional
            The file name for the final output. This file only gives the
            averaged values, rather than individual values fitted within
            batches if there are any. Default is summary_output.csv
        full_output_file : str, optional
            The file name for the full output file. This file gives partial
            results from batches, rather than the averaged results. Default is
            full_output.csv
        lightcurve_folder : str, optional
            The folder to save the fitted lightcurves to. Default is
            ./fitted_lightcurves
        plot : bool, optional
            If True, will plot the results and save them to plot_folder
        plot_folder : str, optional
            The folder to save fitted plots to. Default is ./plots
        marker_color : str, optional
            A matplotlib color string to determine the color of the data points
            in plots. Default is 'dimgrey'
        line_color : str, optional
            A matplotlib color string to set the color of the best fit transit
            profile in plots. Default is 'black'
        folded_P : float, optional
            If this is done on folded lightcurves, then this is the period they
            are folded to.
        folded_P_err : float, optional
            The error on folded_P
        folded_t0 : float, optional
            If this is done on folded lightcurves, then this is the t0 they
            are folded to.
        folded_t0_err : float, optional
            The error on folded_t0

        '''
        n_batches = len(results)

        folded = folded_P is None

        def initialise_dict_entry(d, param):
            '''
            Initialises param in results dictionaries
            '''
            if param not in d:
                d[param] = self._full_prior.priors[param].generate_blank_ParamArray()
            return d

        # We pull out the results for all the variables into two dictionaries
        values = {}
        errors = {}

        for i, ri in enumerate(results):
            # Loop through and populate the dictionaries
            # We have to deal with global, filter-specific, and lightcurve-
            # specific parameters slightly differently

            # FIRST: save the detrended curves and the fit:
            io.save_final_light_curves(lightcurves[i], priors[i], results[i], lightcurve_folder, folded)

            # Now plot the curves!
            if plot:
                try:
                    plot_individual_lightcurves(lightcurves[i], priors[i],
                                                results[i], folder_path=plot_folder,
                                                marker_color=marker_color,
                                                line_color=line_color)
                except Exception as e:
                    print('Exception raised while plotting:')
                    print(e)
                    print('Traceback:')
                    traceback.print_tb(e.__traceback__)

            # We will pull things out of the prior info
            for j, param_info in enumerate(priors[i].fitting_params):
                param_name, batch_tidx, batch_fidx, batch_eidx = param_info

                # The fidx, tidx, eidx in param_info are for the batch. We want
                # the global values, so pull out of the lightcurves

                # for global params, these are all None
                if param_name in global_params or (param_name == 't0' and not priors[i].fit_ttv):
                    tidx, fidx, eidx = None, None, None

                # For filter dependent parameters, batch_tidx and
                # batch_eidx are None, so we need to go through the
                # lightcurves to find the filter index
                elif param_name in filter_dependent_params:
                    # Find the filter index
                    for k in np.ndindex(lightcurves[i].shape):
                        if k[1] == batch_fidx and lightcurves[i][k] is not None:
                            tidx = None
                            fidx = lightcurves[i][k].filter_idx
                            eidx = None
                            break
                else:
                    # Lightcurve specific, we can just pull it straight out.
                    tidx = lightcurves[i][batch_tidx, batch_fidx, batch_eidx].telescope_idx
                    fidx = lightcurves[i][batch_tidx, batch_fidx, batch_eidx].filter_idx
                    eidx = lightcurves[i][batch_tidx, batch_fidx, batch_eidx].epoch_idx

                # Check that the parameter has been initialised
                values = initialise_dict_entry(values, param_name)
                errors = initialise_dict_entry(errors, param_name)

                if values[param_name][tidx, fidx, eidx] is None:
                    values[param_name][tidx, fidx, eidx] = []
                if errors[param_name][tidx, fidx, eidx] is None:
                    errors[param_name][tidx, fidx, eidx] = []

                # Now pull out the best values
                values[param_name][tidx, fidx, eidx].append(ri.best[j])
                errors[param_name][tidx, fidx, eidx].append(ri.uncertainties[j])

        # Now we have collated all of the results from each run, we can take
        # weighted averages to get the final values for each parameter
        best_vals = {}
        best_vals_errors = {}

        for param in values:
            # Loop through all the parameters
            best_vals = initialise_dict_entry(best_vals, param)
            best_vals_errors = initialise_dict_entry(best_vals_errors, param)

            for i in np.ndindex(values[param].shape):
                if values[param][i] is not None:
                    # Get the weighted average and error
                    val, err = weighted_avg_and_std(values[param][i], errors[param][i])
                    best_vals[param][i] = val
                    best_vals_errors[param][i] = err

        if fit_ld:
            # We have to deal here with the limb darkening
            # Since we fit for the Kipping q parameters, we should give
            # these and the physical values (denoted as u)
            # We put these in both the values and best_vals arrays

            # Intialise entries
            u_coeffs = []
            for param in self.ld_coeffs:
                # Initialise each of the u params
                ldc_q = 'q{}'.format(param[-1])
                ldc_u = 'u{}'.format(param[-1])
                u_coeffs.append(ldc_u)

                if ldc_u not in best_vals:
                    best_vals[ldc_u] = values[ldc_q].generate_blank_ParamArray()
                    values[ldc_u] = values[ldc_q].generate_blank_ParamArray()
                if ldc_u not in best_vals_errors:
                    best_vals_errors[ldc_u] = errors[ldc_q].generate_blank_ParamArray()
                    errors[ldc_u] = errors[ldc_q].generate_blank_ParamArray()


            for i in np.ndindex(values['q0'].shape):
                if values['q0'][i] is not None:
                    # Initialise empty lists
                    for u in u_coeffs:
                        values[u][i] = []
                        errors[u][i] = []

                    # Put all the q values for a given filter into one place so we
                    # can access q0, q1 simultaneously for converting to u
                    # All values
                    filter_q = np.vstack((values[q][i] for q in self.ld_coeffs)).T
                    filter_q_errors = np.vstack((errors[q][i] for q in self.ld_coeffs)).T

                    # Best (averaged) values
                    best_filter_q = np.vstack((best_vals[q][i] for q in self.ld_coeffs)).T[0]
                    best_filter_q_errors = np.vstack((best_vals_errors[q][i] for q in self.ld_coeffs)).T[0]

                    # First do all values
                    for j, qj in enumerate(filter_q):
                        u, u_err = self._full_prior.ld_handler.convert_qtou_with_errors(qj, filter_q_errors[j], self.limb_darkening_model)

                        for k, uk in enumerate(u_coeffs):
                            values[uk][i].append(u[k])
                            errors[uk][i].append(u_err[k])

                    # Now do best values
                    u, u_err = self._full_prior.ld_handler.convert_qtou_with_errors(best_filter_q, best_filter_q_errors, self.limb_darkening_model)
                    for k, uk in enumerate(u_coeffs):
                        best_vals[uk][i] = u[k]
                        best_vals_errors[uk][i] = u_err[k]

        # TODO - Detrending:
        # Sometimes detrending is done with batches, and this is outputting
        # fits for detrended lightcurves. We want to pull out the detrending
        # parameter fit results from the runs where detrending took place and
        # put them in to the summary (best values)

        # We will have two output files:
        # - summary, which gives the averaged values
        # - full, which gives all values
        # These have been separated for simplicity
        summary_dict = []
        full_dict = []

        for param in values:
            # Loop through each parameter

            # Get the values and errors
            for i in np.ndindex(values[param].shape):
                if values[param][i] is not None:
                    # Get the summary/full values and errors
                    summary_val = best_vals[param][i]
                    summary_err = best_vals_errors[param][i]

                    full_val = values[param][i]
                    full_err = errors[param][i]

                    # Now we need to sort out the display name of the parameter
                    # so we include the telescope, filter, and epoch indices
                    if param in global_params:
                        telescope_idx = '-'
                        filter_idx = '-'
                        epoch_idx = '-'
                    elif param in filter_dependent_params:
                        telescope_idx = '-'
                        filter_idx = i[1]
                        epoch_idx = '-'
                    else:
                        telescope_idx = i[0]
                        filter_idx = i[1]
                        epoch_idx = i[2]

                    # Add the results to the output dicts
                    summary_dict.append({'Parameter' : param,
                                         'Telescope' : telescope_idx,
                                         'Filter' : filter_idx,
                                         'Epoch' : epoch_idx,
                                         'Best value' : summary_val,
                                         'Uncertainty' : summary_err})

                    for j in range(len(full_val)):
                        # We have to loop over the batches for the full results
                        full_dict.append({'Parameter' : param,
                                          'Telescope' : telescope_idx,
                                          'Filter' : filter_idx,
                                          'Epoch' : epoch_idx,
                                          'Batch' : j,
                                          'Best value' : full_val[j],
                                          'Uncertainty' : full_err[j]})



        # Save the outputs!
        # First the full one:
        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, full_output_file), 'w') as f:
            columns = ['Parameter', 'Telescope', 'Filter', 'Epoch', 'Batch',
                       'Best value', 'Uncertainty']

            writer = csv.DictWriter(f, columns)
            writer.writeheader()
            writer.writerows(full_dict)

        # Now the summary file
        with open(os.path.join(output_folder, summary_file), 'w') as f:
            columns = ['Parameter', 'Telescope', 'Filter', 'Epoch',
                       'Best value', 'Uncertainty']

            writer = csv.DictWriter(f, columns)
            writer.writeheader()
            writer.writerows(summary_dict)

        return summary_dict, full_dict

    ##########################################################
    #             UTILITY FUNCTIONS                          #
    ##########################################################
    def _calculate_n_params(self, lightcurves, indices, ld_fit_method, normalise, detrend):
        '''
        Calculates the number of parameters which would be fitted for a
        given set of filter and epoch indices

        This function exists because it's much faster than repeatedly making
        PriorInfos for different combos.

        Parameters
        ----------
        indices : tuple
            The tuple of indices to consider. Must be given as
            (telescope_indices, filter_indices, epoch_indices)

        '''
        indices = self._format_indices(indices)

        lightcurve_subset = self._get_lightcurve_subset(lightcurves, indices)
        detrending_indices = self._get_detrending_subset(indices)

        unique_indices = self._get_unique_indices(indices)

        n_filters, n_epochs = (len(unique_indices[1]), len(unique_indices[2]))

        n_lightcurves = (lightcurve_subset != None).sum()

        # Account for global parameters
        n_params = self.n_global_params

        # Account for filter-specific parameters - rp and LD coeffs
        if ld_fit_method in ['independent', 'coupled']:
            n_params += n_filters * (1 + self.n_ld_params)
        else: # single fitting mode being used
            n_params += n_filters + self.n_ld_params

        # Account for normalisation
        if normalise:
            n_params += n_lightcurves

        # Account for detrending
        if detrend:
            for i in np.array(indices).T:
                subset_i = self._full_to_subset_index(indices, i)
                if lightcurve_subset[subset_i][0] is not None:
                    detrending_index = detrending_indices[subset_i][0]
                    detrending_info = self.detrending_info[detrending_index]
                    detrending_model = self.detrending_models[detrending_index]
                    if detrending_info[0] in ['nth order', 'custom']:
                        n_params += detrending_model.n_params
                    elif detrending_info[0] == 'off':
                        pass
                    else:
                        raise ValueError('Unrecognised detrending model {}'.format(detrending_info[0]))

        return n_params

    def _format_indices(self, indices):
        '''
        If passed a set of indices, checks they are usable.
        If indices is None, sets them to cover all of the possible values
        '''
        if indices is None:
            return tuple(np.array(list(np.ndindex(self.all_lightcurves.shape))).T)

        return indices

    def _get_unique_indices(self, indices):
        '''
        When given a tuple of indices of all light curves to consider,
        gets all the unique values
        '''
        return [np.unique(i) for i in indices]

    def _full_to_subset_index(self, subset_indices, full_index):
        '''
        Converts an index which uses the notation of full parameter space to
        a subset. USeful for converting between overall indexing and indexing
        within a batch

        Parameters
        ----------
        subset_indices : tuple
            The indices which define the full subset of light curves
        full_index : array_like, shape (3, )
            The full-notation index to be converted
        '''
        unique_indices = self._get_unique_indices(subset_indices)

        return tuple((np.where(unique_indices[i] == full_index[i])[0]) for i in range(len(full_index)))

    def _subset_to_full_index(self, subset_indices, subset_index):
        '''
        Converts an index from notation within a batch to the full indexing.
        Inverse of _full_to_subset_index

        Parameters
        ----------
        subset_indices : tuple
            The indices which define the full subset of light curves
        subset_index : array_like, shape (3, )
            The subset-notation index to be converted
        '''
        unique_indices = self._get_unique_indices(subset_indices)

        return unique_indices[subset_index]
