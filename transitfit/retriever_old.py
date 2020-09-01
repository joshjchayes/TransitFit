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
from ._utils import get_normalised_weights, get_covariance_matrix, validate_lightcurve_array_format, weighted_avg_and_std
from . import io
from .plotting import plot_individual_lightcurves
from .lightcurve import LightCurve
from .detrending_funcs import NthOrderDetrendingFunction
from .detrender import DetrendingFunction

import numpy as np
from dynesty import NestedSampler
import os
import csv
from copy import deepcopy
import multiprocessing as mp

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

        self.limb_darkening_model = limb_darkening_model
        self.host_T = host_T
        self.host_logg = host_logg
        self.host_z = host_z
        self.host_r = host_r
        self.ldtk_cache = ldtk_cache
        self.n_telescopes = n_telescopes
        self.n_filters = n_filters
        self.n_epochs = n_epochs
        self.n_ld_samples = n_ld_samples
        self.do_ld_mc = do_ld_mc
        self.fit_ttv = fit_ttv

        # Read in the filters
        if self._filter_input is None:
            self.filters = None
        elif type(self._filter_input) == str:
            self.filters = read_filter_info(self._filter_input)
        else:
            self.filters = parse_filter_list(self._filter_input)

        self.all_lightcurves, self.detrending_index_array = read_input_file(data_files, data_skiprows)

        self.n_total_lightcurves = np.sum(self.all_lightcurves!=None)

        # Make the all-encompassing prior to pull out some info from
        # Assume independent ld method - it doesn't actually matter
        # since this never gets used.
        self._full_prior, _ = self._get_priors_and_curves('independent')

        self.n_global_params = 0
        for param in global_params:
            if param in self._full_prior.fitting_params:
                self.n_global_params += 1

        # Get the number of parameters in the limb darkening
        self.n_ld_params = len(self._full_prior.limb_dark_coeffs)
        self.ld_coeffs = self._full_prior.limb_dark_coeffs

    def run_retrieval(self, ld_fit_method='independent', fitting_mode='auto',
                      max_parameters=25, maxiter=None, maxcall=None,
                      sample='auto', nlive=300, dlogz=None, plot_final=True,
                      plot_partial=True,
                      results_output_folder='./output_parameters',
                      final_lightcurve_folder='./fitted_lightcurves',
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

        if fitting_mode.lower() == 'auto':
            # Generate the basic PriorInfo, needed to calculate how many
            # parameters would be fitted
            n_params_for_complete = self._calculate_n_params(None, ld_fit_method,
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
            print('Beginning "all" mode fitting')
            # We are fitting everything simultaneously
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

            self.save_batched_results([results], [priors], [lightcurves], output_folder='./output_parameters', fit_ld=ld_fit_method!='off')

            return results

        elif fitting_mode.lower() == 'folded':
            # Calculate the fitting batches
            batches = self.get_batches_for_folding(max_parameters, ld_fit_method,
                                              detrend, normalise, overlap)

            # work out how many batches we have
            n_batches = sum([len(filter) for filter in batches])
            print('Beginning fitting of {} batches'.format(n_batches))

            # Blank list to fill with results
            results_list = [[] for i in range(self.n_filters)]
            priorinfo_list = [[] for i in range(self.n_filters)]
            lightcurve_list = [[] for i in range(self.n_filters)]

            # Now go through each batch and run retrieval, saving results to
            # results_list
            for fi, filter_batches in enumerate(batches):
                for bi, batch in enumerate(filter_batches):
                    print('Fitting batch {} of {} for filter {} of {}'.format(bi+1, len(filter_batches), fi + 1, len(batches)))

                    # Set up priors and lightcurves
                    priors, lightcurves = self._get_priors_and_curves(ld_fit_method,
                                                      indices=batch,
                                                      detrend=detrend,
                                                      normalise=normalise)

                    priorinfo_list[fi].append(priors)
                    lightcurve_list[fi].append(lightcurves)

                    # Run retrieval!
                    results, ndof = self._run_dynesty(lightcurves, priors,
                                                      maxiter, maxcall, sample,
                                                      nlive, dlogz)

                    results_list[fi].append(results)

                # Save output
                _, _ = self.save_batched_results(results_list[fi], priorinfo_list[fi], lightcurve_list[fi], output_folder='./output_parameters', summary_file='filter_{}_summary.csv'.format(fi), full_output_file='filter_{}_full_output.csv'.format(fi), lightcurve_folder='./fitted_lightcurves/filter-fitted_curves/filter_{}'.format(fi), plot_folder='./plots/filter-fitted_curves/filter_{}'.format(fi), fit_ld=ld_fit_method!='off')

            # Once we have run all the batches, we need to combine the results
            # to produce some folded lightcurves for each filter, and then
            # to run fitting across the wavelengths.
            # For global parameters P and t0, we won't be fitting these, but
            # will instead use a weighted mean from all the individual filter
            # fits.
            folded_curves, best_P, best_P_err, best_t0, best_t0_err = self._fold_lightcurves(results_list, priorinfo_list, lightcurve_list)

            folded_batches = self.get_filter_batches(folded_curves, max_parameters, ld_fit_method,
                                                     False, False, overlap)

            folded_results = []
            folded_priors = []
            folded_lightcurves = []

            for bi, batch in enumerate(folded_batches):
                # Now we need to make the priors for the folded lightcurves
                # Note that we are not fitting for period or t0, but are still
                # fitting a, w, ecc, inc etc

                filters_in_batch = np.unique(batch[1])

                priors, _ = self._get_priors_and_curves(ld_fit_method, batch,
                                                  detrend=False, normalise=False,
                                                  folded=True, folded_P=best_P,
                                                  folded_t0=best_t0)
                folded_priors.append(priors)

                # Now we get the relevant folded lightcurves
                lightcurves = folded_curves[:,filters_in_batch,:].reshape(1, len(filters_in_batch), 1)

                folded_lightcurves.append(lightcurves)

                results, ndof = self._run_dynesty(lightcurves, priors, maxiter,
                                                 maxcall, sample, nlive, dlogz)

                folded_results.append(results)

            #return folded_results, folded_priors, folded_lightcurves

            # Make outputs etc
            full_result, summary_result = self.save_batched_results(folded_results, folded_priors, folded_lightcurves, lightcurve_folder='./fitted_lightcurves/final_curves', plot_folder='./plots/final_folded_lightcurves', fit_ld=ld_fit_method!='off')

            return folded_results

        elif fitting_mode.lower() == 'batched':
            # We aren't folding, but we still want to not run everything
            # simultaneously

            # First get the batches
            batches = self.get_filter_batches(self.all_lightcurves, max_parameters, ld_fit_method,
                                                     True, True, overlap)

            all_results = []
            all_priors = []
            all_lightcurves = []

            for bi, batch in enumerate(batches):
                filters_in_batch = np.unique(batch[1])
                # Now we want to get the lightcurves and priors for each batch
                priors, lightcurves = self._get_priors_and_curves(ld_fit_method, batch)

                results, ndof = self._run_dynesty(lightcurves, priors, maxiter,
                                                  maxcall, sample, nlive,
                                                  dlogz)

                all_results.append(results)
                all_priors.append(priors)
                all_lightcurves.append(lightcurves)

            # Make outputs etc
            full_result, summary_result = self.save_batched_results(all_results, all_priors, all_lightcurves, lightcurve_folder='./fitted_lightcurves/final_curves', fit_ld=ld_fit_method!='off')

            return all_results

        else:
            raise ValueError('Unrecognised fitting mode {}'.format(fitting_mode))

    def _get_priors_and_curves(self, ld_fit_method, indices=None,
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
            are folded to

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

            lightcurves = None
            n_telescopes = 1
            n_filters = len(filter_indices)
            n_epochs = 1
            detrend = False
            normalise = False

        else:
            # get the lightcurve and detrending index array into the right shapes
            # and extract the relevant info.
            lightcurves, detrending_indices = self._get_curves_and_detrending(indices)

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
                                      self.fit_ttv, lightcurves)
        else:
            # Reading in from a list
            priors = parse_priors_list(self._prior_input,
                                       n_telescopes,
                                       n_filters,
                                       n_epochs,
                                       self.limb_darkening_model,
                                       filter_indices,
                                       folded, folded_P, folded_t0,
                                       self.host_r, self.fit_ttv, lightcurves)

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
            priors.fit_detrending(lightcurves,
                                  self.detrending_info, detrending_indices)

        # Set up normalisation
        if normalise:
            priors.fit_normalisation(lightcurves)

        return priors, lightcurves

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
        # test having a deepcopy thing here????
        lightcurves = validate_lightcurve_array_format(lightcurves)

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
                return ln_likelihood + priors.ld_handler.ldtk_lnlike(np.array(u).T, limb_dark)
            else:
                return ln_likelihood
        #######################################################################
        #######################################################################

        # Now we can set up and run the sampler!
        sampler = NestedSampler(lnlike, prior_transform, n_dims, bound='multi',
                                sample=sample, #update_interval=float(n_dims),
                                nlive=nlive)

        try:
            print('Beginning retrieval of {} parameters'.format(n_dims))
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

    def _get_curves_and_detrending(self, indices, set_detrending=False,
                                   set_normalisation=False):
        '''
        Returns the light curves and detrending indices for the given
        telescope, filter and epoch indices (given as tuple).

        Parameters:
        -----------
        indices :

        set_detrending : bool, optional
            If True, will initialise detrending for the lightcurves. Default is
            False
        set_normalisation : bool, optional
            If True, will initialise normalisation for the lightcurves. Default
            is False
        '''
        # Get the unique indices
        unique_indices = self._get_unique_indices(indices)
        # Make some empty arrays for us to populate
        lightcurves = np.full(tuple(len(idx) for idx in unique_indices), None)
        detrending_indices = np.full(tuple(len(idx) for idx in unique_indices), None)

        # Go through each index being used and put in the relevant info
        # deepcopy is used here to ensure we don't end up with clashing
        # attributes (e.g. from detrending twice)
        for index in np.array(indices).T:
            subset_index = self._full_to_subset_index(indices, index)
            lightcurves[subset_index] = deepcopy(self.all_lightcurves[tuple(index)])
            detrending_indices[subset_index] = deepcopy(self.detrending_index_array[tuple(index)])

            if set_detrending:
                detrending_index = detrending_indices[subset_index]
                model = self.detrending_info[detrending_index][0]
                if not model == 'off':
                    if model == 'nth order':
                        order = self.detrending_info[detrending_index][1]
                        lightcurves[subset_index].set_detrending(model, order=order)
                    elif mode == 'custom':
                        function = self.detrending_info[detrending_index][1]
                        lightcurves[subset_index].set_detrending(model, function=function)
                    else:
                        print('WARNING: unrecognised detrending method {}. Detrending for lightcurve {} not set'.format(subset_index))

            if set_normalisation:
                lightcurves[subset_index].set_normalisation()

        return lightcurves, detrending_indices

    def _calculate_n_params(self, indices, ld_fit_method, normalise, detrend):
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

        lightcurves, detrending_indices = self._get_curves_and_detrending(indices)

        unique_indices = self._get_unique_indices(indices)

        n_filters, n_epochs = (len(unique_indices[1]), len(unique_indices[2]))

        n_lightcurves = (lightcurves != None).sum()

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
                if lightcurves[subset_i][0] is not None:
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

    def _get_unique_indices(self, indices):
        '''
        When given a tuple of indices of all light curves to consider,
        gets all the unique values
        '''
        return [np.unique(i) for i in indices]

    def _format_indices(self, indices):
        '''
        If passed a set of indices, checks they are usable.
        If indices is None, sets them to cover all of the possible values
        '''
        if indices is None:
            return tuple(np.array(list(np.ndindex(self.all_lightcurves.shape))).T)

        return indices

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
        batches : array_like, shape (n_filters, )
            The batches. Each entry should be a list of indices.
        lightcurves : array_like, shape (n_filters, )
            The lightcurves. Each entry should be a list of LightCurve arrays
            used for the batches within each filter

        Returns
        -------
        folded_lightcurves : np.array, shape (1, n_filters, 1)
            All the lightcurves, with each filter folded onto one epoch.
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
                    print(i)
                    tidx, fidx, eidx = i
                    retrieved_P.append(results_dict['P'][i])
                    retrieved_P_err.append(errors_dict['P'][i])

                    retrieved_t0[eidx].append(results_dict['t0'][i])
                    retrieved_t0_err[eidx].append(errors_dict['t0'][i])

        # Find the weighted average to get the best fit values of P and t0
        best_P, P_err = weighted_avg_and_std(retrieved_P, retrieved_P_err)
        best_t0, t0_err = weighted_avg_and_std(retrieved_t0, retrieved_t0_err)

        print(best_t0, t0_err)

        if not self.fit_ttv:
            # Put the t0 values and errors into an array just for simplicity
            # in dealing with the two modes
            best_t0 = np.array([best_t0]).reshape(1,1)
            t0_err = np.array([t0_err]).reshape(1,1)

        print(best_t0, t0_err)

        print('Folding lightcurves')
        print('P = {} ± {}'.format(round(best_P, 8),  round(P_err, 8)))
        for i, t0i in enumerate(best_t0):
            print('t0 = {} ± {} (epoch {})'.format(t0i.round(3)[0],  t0_err[i].round(3)[0], i))



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
                    method_idx = lc.detrending_method_idx
                    # Get the detrending coeffs for this lc
                    detrending_coeffs = prior.detrending_coeffs[method_idx]

                    best_d = [results_dict[d][i] for d in detrending_coeffs]
                    norm = results_dict['norm'][i]

                    # Make the detrended light curve and fold to the final t0

                    batch_detrended_curves.append(lc.create_detrended_LightCurve(best_d, norm))


                #################


                # Make the fitting params a np array
                fitting_params = np.array(prior.fitting_params)

                # Get the best results
                best_results = result.best

                # go through each set of results and create the detrended
                # light curves

                # The best fit detrending and normalisation values.
                # These are stored and then used so that we always detrend
                # and THEN normalise
                detrending = np.full(lcs.shape, None, object)
                normalisation = np.ones(lcs.shape)

                skip = 0 # used for skipping detrending entries
                for pi, p in enumerate(prior.fitting_params):
                    if skip > 0:
                        skip -= 1
                    else:
                        # For this, the retrieved results of interest are
                        # detrending coefficients and normalisation constants.

                        # Get the indices
                        telescope_idx = prior._telescope_idx[pi]
                        filter_idx = prior._filter_idx[pi]
                        epoch_idx = prior._epoch_idx[pi]

                        if p[0] == 'd':
                            # Detrending

                            # We assume that the detrending coeff encountered
                            # is the first one for a given lightcurve

                            # get the number of detrending params
                            n_detrending = lcs[telescope_idx, filter_idx,epoch_idx].n_detrending_params

                            # Get all the detrending coeffs and store
                            detrending[telescope_idx, filter_idx, epoch_idx] = best_results[pi:pi+n_detrending]

                            # Set the number of parameters to skip.
                            skip = n_detrending - 1

                        elif p == 'norm':
                            # Normalisation
                            normalisation[telescope_idx, filter_idx, epoch_idx] = best_results[pi]

                # Now we can normalise and detrend the lightcurves

                # since each batch should only be across one filter, we can
                # combine all the lightcurves into one.

                final_batch_lightcurve = None

                for idx in np.ndindex(lcs.shape):
                    if lcs[idx] is not None:
                        # Detrend/normalise and fold to t0/P. Then combine with
                        # the final curve

                        if final_batch_lightcurve is None:
                            final_batch_lightcurve = lcs[idx].create_detrended_LightCurve(detrending[idx], normalisation[idx]).fold(best_t0, best_P)

                        else:
                            final_batch_lightcurve = lcs[idx].create_detrended_LightCurve(detrending[idx], normalisation[idx]).fold(best_t0, best_P).combine(final_batch_lightcurve, filter_idx=fi)

                # Store the final batched lightcurves
                final_batched_lightcurves[fi].append(final_batch_lightcurve)

        # Now that all batches have been folded, we can go through each
        # filter to produce the final lightcurves!!
        final_curves = []
        for fi, final_batch_curves in enumerate(final_batched_lightcurves):
            if len(final_batch_curves) == 1:
                # Only one batch
                final_curves.append(final_batch_curves[0])
            else:
                # Combine all the final curves together
                final_curves.append(final_batch_curves[0].combine(final_batch_curves[1:], filter_idx=fi))

        final_curves = np.array(final_curves).reshape(1, len(final_curves), 1)

        return final_curves, best_P, P_err, best_t0, t0_err

    def _save_outputs(self, lightcurves, results, priors, result_folder, output_fname,
                      final_lightcurve_base_folder,
                      light_curve_specific_folder,
                      plot_base_folder, plot_specific_folder,
                      marker_color, line_color):
        '''
        Saves the output parameters, final detrended lightcurves and plots
        curves
        '''
        # Save output
        try:
            save_fname = os.path.join(result_folder, output_fname)
            os.makedirs(result_folder, exist_ok=True)
            io.save_results(results, priors, save_fname)
            print('Best fit parameters saved to {}'.format(os.path.abspath(save_fname)))
        except Exception as e:
            print('The following exception was raised whilst saving parameter results:')
            print(e)

        # Save final light curves
        try:
            output_folder = os.path.join(final_lightcurve_base_folder, light_curve_specific_folder)
            os.makedirs(output_folder, exist_ok=True)
            io.save_final_light_curves(lightcurves, priors,
                                       results, output_folder)
            print('Fitted light curves saved to {}'.format(os.path.abspath(output_folder)))
        except Exception as e:
            print('The following exception was raised whilst saving final light curves:')
            print(e)

        # Plot the final curves!
        try:
            save_folder = os.path.join(plot_base_folder, plot_specific_folder)
            os.makedirs(save_folder, exist_ok=True)
            plot_individual_lightcurves(lightcurves, priors,
                                        results, folder_path=save_folder,
                                        marker_color=marker_color,
                                        line_color=line_color, fnames=None)
            print('Plots saved to {}'.format(os.path.abspath(save_folder)))
        except Exception as e:
            # TODO: Try plotting from files rather than results objects
            print('The following exception was raised whilst plotting final light curves:')
            print(e)

    def get_batches_for_folding(self, max_parameters, ld_fit_method,
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
                    n_params = self._calculate_n_params(test_batch, ld_fit_method,
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

        print('Batches for folding calculated:')
        print(all_batches)

        return all_batches

    def get_filter_batches(self, lightcurves, max_parameters, ld_fit_method,
                           detrend, normalise, overlap=2):
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

        Returns
        -------
        batches : array_like, shape (n_batches)
            The final batches. Each entry is a tuple of
            (telescope indices, filter indices, epoch indices) using the
            indices of lightcurves. Each of these batches can then be passed to
            _get_priors_and_curves.
        '''
        # The final batches
        all_batches = []

        n_filters = lightcurves.shape[1]

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
                indices = np.where(self.all_lightcurves[:,fi,:] != None)
                n_curves = len(indices[0])

                # Put the fi index back into indices tuple, converting to list
                indices = (list(indices[0]), list(np.full(n_curves, fi)), list(indices[1]))

                # Make the test batch
                test_batch = tuple((single_batch[j] + indices[j] for j in range(3)))

                # Get the number of params for the test batch
                n_params = self._calculate_n_params(test_batch, ld_fit_method,
                                                    normalise, detrend)

                if n_params <= max_parameters:
                    # Add the filter to the batch
                    single_batch = deepcopy(test_batch)
                    filters_in_batch += 1
                    # Check to see if we have completed the loop and have all
                    # the filters in at least one batch
                    if fi == (n_filters - 1):
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

        print('Filter batches calculated:')
        print(all_batches)

        return all_batches

    def save_batched_results(self, results, priors, lightcurves,
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
            lightcurvearray for the batch
        filepath : str, optional
            The path to save the results to
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

            # We will pull things out of the prior info
            for j, param_info in enumerate(priors[i].fitting_params):
                param_name, batch_tidx, batch_fidx, batch_eidx = param_info

                # The fidx, tidx, eidx in param_info are for the batch. We want
                # the global values, so pull out of the lightcurves

                # for global params, these are all None
                if param_name in global_params or (param_name in 't0' and not priors[i].fit_ttv):
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
