'''
RetrievalSplitter

Object which handles splitting large numbers of light curves into smaller
chunks to run retrieval on. This is used so that the nested sampling retrieval
doesn't encounter crazy numbers of simultaneously fitted parameters and break.
'''

import numpy as np
from copy import deepcopy

from .retriever import Retriever
from .priorinfo import setup_priors


class RetrievalSplitter:
    def __init__(self, lightcurves, priorinfo):
        '''
        RetrievalSplitter deals with large numbers of input curves and splits
        them into multiple retrieval batches.

        Parameters
        ----------
        lightcurves : array_like, shape (n_telescopes, n_filters, n_epochs)
            An array of LightCurves. If no data exists for a point in the array
            then the entry should be `None`.
        priorinfo : PriorInfo
            The priors for fitting. This basically sets the intervals over
            which each parameter is fitted.
        '''

        self.lightcurves = lightcurves
        self.priorinfo = priorinfo

        # Check which global parameters are being used.
        global_fitting_params = ['P', 't0', 'a', 'inc', 'w', 'ecc']

        self.n_global_params = 0
        for param in global_fitting_params:
            if param in self.priorinfo.fitting_params:
                self.n_global_params += 1

    def split_retrieval(self, max_parameters=25, max_subgroup_size=100):
        '''
        Splits the input data into batches for running retrieval.

        Parameters
        ----------
        max_parameters : int, optional
            The maximum number of parameters to be retrieved within each batch.
            Default is 5.
        max_subgroup_size : int, optional
            The maximum number of `LightCurve`s within each batch.
            Default is 100 - Functionally this means that `max_parameters` is
            the controlling variable here.

        Returns
        -------
        lightcurve_batches : list
            A list of new arrays of `LightCurve`s which can be passed to the
            `Retriever`
        priorinfo_batches : list
            The corresponding `PriorInfo` objects for each of the batches.

        Notes
        -----
        The `RetrievalSplitter` will prioritise grouping observations with the
        same filter together, and then try to maximise the time difference
        between the earliest and latest epoch in each batch.
        '''

        # First check if any filters have too many lightcurves to fit in the
        # specified limits for subgroup size and parameters. Warn the user.
        warning_filters = []

        # Also keep track of how many light curves and parameters there are per
        # filter - this will be useful later for combining batches.
        n_curves_in_filter = np.zeros(self.lightcurves.shape[1], int)
        n_params_in_filter = np.zeros(self.lightcurves.shape[1], int)

        # This is the list of original filter indices. We will be combining
        # the different filters into
        original_groups = []

        for i in range(self.priorinfo.n_filters):
            n_curves_in_filter[i] = get_n_curves_in_filter(self.lightcurves, i)
            n_params_in_filter[i] = self.get_n_parameters(n_curves_in_filter[i], 1)

            if n_curves_in_filter[i] > max_subgroup_size or n_params_in_filter[i] > max_parameters:
                warning_filters.append(i)

            original_groups.append([i])

        if not warning_filters == []:
            print('Warning: The following filters have too many light curves to all fit the curves for each filter simultaneously:')
            print(warning_filters)
            print('Consider changing max_parameters or max_subgroup_size, or proceed with caution.')

        # Sort the original groups into decending order of size
        sorted_group_size = np.argsort(n_curves_in_filter)[::-1]


        sorted_original_groups = [original_groups[idx] for idx in sorted_group_size]

        print('Calculating batches')

        # This is what we will slowly change into the final batches.
        batches = []

        for fi, filter_idx in enumerate(sorted_original_groups):
            #print(batches, filter_idx)
            if not filter_in_batches(batches, filter_idx):
                # the filter hasn't been added to a batch yet
                # The batch for testing
                filter_set = filter_idx

                for fj, test_filter in enumerate(sorted_original_groups[fi + 1:]):
                    # Loop over all smaller groups
                    if not filter_in_batches(batches, test_filter):
                        # The test filter hasn't been added to a batch
                        test_set = filter_set + test_filter

                        # How many curves and parameters would this batch need?
                        n_curves = sum(n_curves_in_filter[test_set])
                        n_params = self.get_n_parameters(n_curves, len(test_set))

                        if n_curves <= max_subgroup_size and n_params <= max_parameters:
                            # If the batch is within limits, update the set!
                            filter_set = deepcopy(test_set)

                # Now we have a maximally-full filter set!
                # Add the filter set as a batch
                batches.append(filter_set)

        print('Batches have been set.')
        print('Filter batches are:', batches)

        # Now we have the batches, we need to split up the original inputs into
        # the batches. This involves splitting the lightcurve inputs and also
        # creating new PriorInfos for each batch
        print('Splitting light curves into batches...')
        batched_lightcurves = self._split_lightcurves(batches)

        print('Preparing PriorInfo for batches...')
        batched_priorinfos = self._split_priorinfos(batches, batched_lightcurves)


        print('All prepared')
        return batched_lightcurves, batched_priorinfos, batches

    def get_n_parameters(self, n_curves, n_filters):
        '''
        When given a number of `LightCurve`s and filters, will calculate the
        number of parameters which would be simultaneously fitted

        Parameters
        ----------
        n_curves : int
            The number of `LightCurve`s
        n_filters : int
            The number of filters

        Returns
        -------
        n_parameters : int
            The number of parameters which would be fitted simultaneously
        '''
        return self.n_global_params + n_filters * (1 + len(self.priorinfo.limb_dark_coeffs)) + n_curves * (1 + len(self.priorinfo.detrending_coeffs))

    def _split_lightcurves(self, batches):
        '''
        When given a set of batches, will split the original light curve input
        into a light curve input for each batch
        '''
        lightcurves = []

        for batch in batches:
            lightcurves.append(self.lightcurves[:,batch,:])

        return lightcurves


    def _split_priorinfos(self, batches, batched_lightcurves):
        '''
        When given a set of batches, will split the original priorinfo input
        into a priorinfo input for each batch
        '''
        priorinfos = []

        for bi, batch in enumerate(batches):
            # Go through and create new PriorInfos with only info on the
            # filters in each batch

            # Set up the basic priorinfo
            priorinfo = setup_priors(self.priorinfo.priors['P'].default_value,
                                     self.priorinfo.priors['rp'][tuple([i[0] for i in np.where(self.priorinfo.priors != None)])].default_value,
                                     self.priorinfo.priors['a'].default_value,
                                     self.priorinfo.priors['inc'].default_value,
                                     self.priorinfo.priors['t0'].default_value,
                                     self.priorinfo.priors['ecc'].default_value,
                                     self.priorinfo.priors['w'].default_value,
                                     self.priorinfo.limb_dark,
                                     self.priorinfo.n_telescopes,
                                     len(batch),
                                     self.priorinfo.n_epochs)

            # Now we need to go through each of the fitting parameters, and
            # add them if they are applicable to the filter
            for pi, param in enumerate(self.priorinfo.fitting_params):
                # We want to consider global/filter parameters first.
                # light curve specific things (detrending etc) will be added
                # later, as will limb darkening, which has to be initialised
                # properly
                if param in ['P', 'a', 'inc', 'ecc', 't0', 'w']:
                    priorinfo.add_uniform_fit_param(param,
                                                    self.priorinfo.priors[param].default_value,
                                                    self.priorinfo.priors[param].low_lim,
                                                    self.priorinfo.priors[param].high_lim,
                                                    self.priorinfo._telescope_idx[pi],
                                                    self.priorinfo._filter_idx[pi],
                                                    self.priorinfo._epoch_idx[pi])
                if param in ['rp']:
                    idx = self.priorinfo._filter_idx[pi]
                    if idx in batch:
                        # Check to see that this is a filter within the batch
                        # We need to work out what the filter index is within
                        # THIS prior info
                        filter_idx = np.where(np.array(batch) == idx)[0][0]
                        priorinfo.add_uniform_fit_param(param,
                                                        self.priorinfo.priors[param][idx].default_value,
                                                        self.priorinfo.priors[param][idx].low_lim,
                                                        self.priorinfo.priors[param][idx].high_lim,
                                                        self.priorinfo._telescope_idx[pi],
                                                        filter_idx,
                                                        self.priorinfo._epoch_idx[pi])

            if self.priorinfo.fit_ld:
                # deal with limb darkening here
                priorinfo.fit_limb_darkening(self.priorinfo.ld_fit_method,
                                             self.priorinfo.ld_handler.host_T,
                                             self.priorinfo.ld_handler.host_logg,
                                             self.priorinfo.ld_handler.host_z,
                                             self.priorinfo.filters[batch],
                                             self.priorinfo._n_ld_samples,
                                             self.priorinfo._do_ld_mc,
                                             self.priorinfo._ld_cache_path)

            if self.priorinfo.detrend:
                # Deal with detrending
                priorinfo.fit_detrending(batched_lightcurves[bi],
                                         self.priorinfo._detrend_method_list,
                                         self.priorinfo._detrend_method_index_array[:,batch,:])


            if self.priorinfo.normalise:
                # Deal with normalisation
                priorinfo.fit_normalisation(batched_lightcurves[bi])

            priorinfos.append(priorinfo)

        return priorinfos


###############################################################################
### UTILITY FUNCTIONS                                                       ###
###############################################################################

def get_n_curves_in_filter(lightcurves, filter_idx):
    '''
    Finds the number of LightCurves for a given filter within an array.

    Parameters
    ----------
    lightcurves : array_like, shape (n_telescopes, n_filters, n_epochs)
        An array of LightCurves. If no data exists for a point in the array
        then the entry should be `None`.
    filter_idx : int
        The filter to consider

    Returns
    -------
    n_curves : int
        The number of non-`None` entries in `lightcurves`
    '''
    return (lightcurves[:,filter_idx,:] != None).sum()

def filter_in_batches(batches, filter_idx):
    '''
    Checks if a given filter is in a batch list

    Parameters
    ----------
    batches : list of lists
        The current batches
    filter_idx : int
        The filter index to consider

    Returns
    -------
    filter_in_batches : bool
    '''
    if batches == []:
        # Empty batches
        return False
    for batch in batches:
        # Go through each batch and see if the filter has been used
        if filter_idx[0] in batch:
            return True
    return False
