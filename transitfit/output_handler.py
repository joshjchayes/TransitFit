'''
Object to deal with writing fitting results to files
'''
import numpy as np
import os
import csv
import pandas as pd
import batman
import itertools
import traceback
import corner
import pickle
from collections.abc import Iterable


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import to_rgb
from matplotlib import colors

from .retriever import global_params, filter_dependent_params, lightcurve_dependent_params
from ._utils import weighted_avg_and_std, host_radii_to_AU
from ._paramarray import ParamArray
from .lightcurve import LightCurve


class OutputHandler:
    '''
    Designed to handle writing outputs from retrieval to files

    Parameters
    ----------
    lightcurves : array_like, shape (n_telescopes, n_filters, n_epochs)
        The full lightcurves array to be retrieved.
    full_prior : PriorInfo
        The prior for the complete light curve dataset.

    '''
    def __init__(self, lightcurves, full_prior, host_r=None):

        self.all_lightcurves = lightcurves

        self.full_prior = full_prior

        self.ld_coeffs = self.full_prior.limb_dark_coeffs
        self.ld_coeffs_u = ['u{}'.format(i) for i in range(len(self.ld_coeffs))]
        self.ld_model = self.full_prior.limb_dark

        self.host_r = host_r

        self.best_model = None
        self.batman_initialised = False

        self.global_params = []
        for i, param in enumerate(self.full_prior.fitting_params):
            if np.all(param[1:] == None):
                self.global_params.append(param[0])

    def save_final_light_curves(self, all_lightcurves, global_prior,
                                folder='./final_light_curves', folded=False):
        '''
        Saves the final curves and best fit model to file

        Requires best model to be initialised

        Parameters
        ----------
        all_lightcurves : array_like
            Array containing the global set of light curves - these should be
            raw and not normalised or detrended.
        '''
        print('Saving final light curves')
        # Set up the model
        self._initialise_batman(all_lightcurves)

        # Put all the detrending coeffs in usable format
        d = np.full(all_lightcurves.shape, None, object)

        if global_prior.detrend:
            # We need to combine the detrending coeff arrays into one
            # Each entry should be a list containing all the detrending
            # coefficients for the light curve.
            for i in np.ndindex(d.shape):
                for coeff in np.ravel(global_prior.detrending_coeffs):
                    if self.best_model[coeff][i] is not None:
                        if d[i] is None:
                            d[i] = [self.best_model[coeff][i][0]]
                        else:
                            d[i].append(self.best_model[coeff][i][0])

        for i, lc in np.ndenumerate(all_lightcurves):
            # Loop through each light curve, make the best model, and save it!
            if lc is not None:
                # First, detrend and normalise the curve


                flux, flux_err = lc.detrend_flux(d[i], self.best_model['norm'][i][0], force_normalise=True)

                # Get phase
                phase = lc.get_phases(self.best_model['t0'][i][0], self.best_model['P'][i][0])

                # Get the best fit model depths
                model_curve = self.batman_models[i].light_curve(self.batman_params[i])

                write_data = np.vstack((lc.times, phase, flux, flux_err, model_curve)).T

                write_data_frame = pd.DataFrame(write_data, columns=['Time', 'Phase', 'Normalised flux', 'Flux uncertainty', 'Best fit curve'])

                save_path = os.path.join(folder,'t{}_f{}_e{}_detrended.csv'.format(*i))

                os.makedirs(folder, exist_ok=True)
                write_data_frame.to_csv(save_path, index=False, na_rep='-')

    def plot_final_light_curves(self, all_lightcurves, global_prior,
                                folder='./plots', figsize=(12,8),
                                marker_color='dimgrey', line_color='black',
                                plot_folded=True, titles=False, bin_data=True,
                                cadence=2, binned_color='red'):
        '''
        Plots the detrended light curves with the global best-fit model

        Parameters
        ----------
        cadence : float, optional
            The cadence to bin to in minutes

        '''
        print('Plotting final curves')
        # First, deal with detrending
        # Put all the detrending coeffs in usable format
        d = np.full(all_lightcurves.shape, None, object)

        if global_prior.detrend:
            # We need to combine the detrending coeff arrays into one
            # Each entry should be a list containing all the detrending
            # coefficients for the light curve.
            for i in np.ndindex(d.shape):
                for coeff in np.ravel(global_prior.detrending_coeffs):
                    if self.best_model[coeff][i] is not None:
                        if d[i] is None:
                            d[i] = [self.best_model[coeff][i][0]]
                        else:
                            d[i].append(self.best_model[coeff][i][0])

        # Now we loop through each curve and make the plots
        for i, lc in np.ndenumerate(all_lightcurves):
            # Loop through each light curve, make the best model, and save it!
            if lc is not None:
                # First, detrend and normalise the curve

                flux, flux_err = lc.detrend_flux(d[i], self.best_model['norm'][i][0], force_normalise=True)

                # Get phase
                phase = lc.get_phases(self.best_model['t0'][i][0], self.best_model['P'][i][0])

                # Get the best fit model depths - use linspaced times for plot
                # and the lc times for residuals
                model_times = np.linspace(lc.times.min(), lc.times.max(), 1000)
                model = batman.TransitModel(self.batman_params[i], model_times)
                model_curve = model.light_curve(self.batman_params[i])
                time_wise_best_curve = self.batman_models[i].light_curve(self.batman_params[i])
                # get model phase:
                n = (model_times - (self.best_model['t0'][i][0] - 0.5 * self.best_model['P'][i][0]))//self.best_model['P'][i][0]

                model_phase = (model_times - self.best_model['t0'][i][0])/self.best_model['P'][i][0] - n + 0.5

                # Get the residual
                residuals = flux - time_wise_best_curve

                plot_errors = [flux_err, None]
                sub_folder = ['with_errorbars', 'without_errorbars']

                for j in range(2):
                    fname = 'individual_curves/{}/t{}_f{}_e{}.png'.format(sub_folder[j],*i)
                    if titles:
                        title = 'Fitted curve: Telescope {} Filter {}, Epoch {}'.format(*i)
                    else:
                        title = None

                    ####### PLOT! #########
                    self._plot_data(phase, flux, plot_errors[j], model_phase,
                                    model_curve, residuals, fname, title, folder,
                                    figsize, marker_color, line_color)

        if not plot_folded:
            return

        # Now we fold the light curves for each filter
        for fi in range(all_lightcurves.shape[1]):
            phase = []
            flux = []
            flux_err = []
            residuals = []
            model_phase = []
            model_flux = []

            for ti, ei in itertools.product(range(all_lightcurves.shape[0]), range(all_lightcurves.shape[2])):
                i = (ti, fi, ei)
                lc = all_lightcurves[i]
                if lc is not None:
                    lc_flux, lc_flux_err = lc.detrend_flux(d[i], self.best_model['norm'][i][0], force_normalise=True)
                    lc_phase = lc.get_phases(self.best_model['t0'][i][0], self.best_model['P'][i][0])

                    # Get the best fit model depths - use linspaced times for plot
                    # and the lc times for residuals
                    model_times = np.linspace(lc.times.min(), lc.times.max(), 1000)
                    model = batman.TransitModel(self.batman_params[i], model_times)
                    lc_model_curve = model.light_curve(self.batman_params[i])
                    time_wise_best_curve = self.batman_models[i].light_curve(self.batman_params[i])

                    # get model phase:
                    n = (model_times - (self.best_model['t0'][i][0] - 0.5 * self.best_model['P'][i][0]))//self.best_model['P'][i][0]

                    lc_model_phase = (model_times - self.best_model['t0'][i][0])/self.best_model['P'][i][0] - n + 0.5

                    lc_residuals = lc_flux - time_wise_best_curve

                    # Store the values!
                    phase += list(lc_phase)
                    flux += list(lc_flux)
                    flux_err += list(lc_flux_err)
                    residuals += list(lc_residuals)
                    model_phase += list(lc_model_phase)
                    model_flux += list(lc_model_curve)

            phase = np.array(phase).flatten()
            flux = np.array(flux).flatten()
            flux_err = np.array(flux_err).flatten()
            residuals = np.array(residuals).flatten()
            model_phase = np.array(model_phase).flatten()
            model_flux = np.array(model_flux).flatten()

            P = self.best_model['P'][None, None, None][0]
            t0 = self.best_model['t0'][None, None, 0][0]

            cadence_days = cadence / (24 * 60)

            cadence_phase = cadence_days/P

            plot_errors = [flux_err, None]
            sub_folder = ['with_errorbars', 'without_errorbars']

            for j in range(2):
                fname = 'folded_curves/{}/filter_{}.png'.format(sub_folder[j],fi)
                if titles:
                    title = 'Folded curve for filter {}'.format(fi)
                else:
                    title = None

                ####### PLOT! #########
                try:
                    self._plot_data(phase, flux, plot_errors[j], model_phase,
                                model_flux, residuals, fname, title, folder,
                                figsize, marker_color, line_color, bin_data,
                                cadence_phase, binned_color)

                except Exception as e:
                    print('Exception raised while plotting:')
                    print(e)
                    print('Traceback:')
                    traceback.print_tb(e.__traceback__)

    def save_complete_results(self, mode, global_prior, output_folder,
                              summary_file):
        '''
        Once all batches etc are run, collates all results and saves to csv
        '''
        _ = self._initialise_best_model(mode, global_prior, output_folder, summary_file)

        print('Saving final results')

        self._save_results_dict(self.best_model, os.path.join(output_folder, 'Complete_results.csv'), False)

    def save_results(self, results, priors, lightcurves,
                     output_folder='./output_parameters',
                     summary_file='summary_output.csv',
                     full_output_file='full_output.csv',
                     samples_plot_folder='./plots', folded=False,
                     plot_posteriors=True, batch_idx=None, stage=1):
        '''
        Saves results to .csv files

        Parameters
        ----------
        results : array_like, shape (n_batches, )
            The results from each run
        priors : array_like, shape (n_batches, )
            The priors for each run
        lightcurves : array_like, shape (n_batches, )
            The light curves for each run
        fit_ld : bool, optional
            Must be true if LDCs are fitted. Default is True
        output_folder : str, optional
            The folder to save output files to (not plots). Default is
            `'./output_parameters'`
        summary_file : str, optional
            The file name for the final output. This file only gives the
            averaged values, rather than individual values fitted within
            batches if there are any. Default is `'summary_output.csv'`
        full_output_file : str, optional
            The file name for the full output file. This file gives partial
            results from batches, rather than the averaged results. Default is
            `'full_output.csv'`

        Returns
        -------
        best_vals : dict
            Each entry is [best val, error]
        combined_dict : dict
            The combined results dictionary. Each entry is a list of values
            from the results dictionaries -
            [[best value, median, 16th percentile, 84th percentile, stdev],...]
        '''
        print('Saving full results...')
        fit_ld = priors[0].fit_ld

        results_dicts = []

        for i, ri in enumerate(results):
            results_dicts.append(self.get_results_dict(ri, priors[i], lightcurves[i]))
            if plot_posteriors:
                try:
                    if folded:
                        sample_folder = os.path.join(samples_plot_folder, 'folded')
                    else:
                        sample_folder = os.path.join(samples_plot_folder, 'unfolded')
                    print(f'Plotting batch {i} samples to {os.path.join(sample_folder, f"batch_{i}_samples.png")}')
                    self._plot_samples(ri, priors[i], f'batch_{i}_samples.png', sample_folder)
                except Exception as e:
                    print(e)
        best_vals, combined_results = self.get_best_vals(results_dicts, fit_ld)

        print(f'Saving summary results to {os.path.join(output_folder, summary_file)}')
        self._save_results_dict(best_vals, os.path.join(output_folder, summary_file), False)
        print(f'Saving full results to {os.path.join(output_folder, full_output_file)}')
        self._save_results_dict(combined_results, os.path.join(output_folder, full_output_file), True)
        print('Results saved')

        return best_vals, combined_results

    def get_results_dict(self, results, priors, lightcurves):
        '''
        Makes dictionaries of results from a single dynesty run.

        Useful for putting results in a form which allows for easy summary etc.

        Parameters
        ----------
        results : dict
            Results from a single dynesty run
        priors : PriorInfo
            The priors for the run
        lightcurves : array_like, shape (n_telescopes, n_filters, n_epochs)
            The lightcurves for the run

        Returns
        -------
        results_dict : dict
            Each entry is [best value, median, 16th percentile, 84th percentile, stdev]
        '''
        print('Extracting results dict')
        results_dict = {}

        # Loop over all fitting parameters and access the results
        for i, param_info in enumerate(priors.fitting_params):
            param_name, batch_tidx, batch_fidx, batch_eidx = param_info

            batch_idx = (batch_tidx, batch_fidx, batch_eidx)
            # GET INDICES
            # The indices here are for a particular batch. We want global
            # values so pull them out of the LightCurves
            full_idx = self._batch_to_full_idx(batch_idx, param_name, lightcurves, priors.allow_ttv)

            result_entry = [results.best[i], results.median[i], results.lower_err[i], results.upper_err[i], results.uncertainties[i]]

            # Check that the parameter has been initialised in the dict
            results_dict = self._initialise_dict_entry(results_dict, param_name)
            if results_dict[param_name][full_idx] is None:
                # Initialise a list
                results_dict[param_name][full_idx] = []

            results_dict[param_name][full_idx].append(result_entry)

        # Go through the PriorInfo and get the constant values out
        # These are ones that were in the priors and NOT in fitting params
        for param_name in priors.priors:
            if param_name not in priors.fitting_params:
                # This is a constant value
                # Check that the parameter has been initialised in the dict
                results_dict = self._initialise_dict_entry(results_dict, param_name)
                for i in np.ndindex(priors.priors[param_name].shape):
                    if priors.priors[param_name][i] is not None:
                        result_entry = [priors.priors[param_name].default_value, None, None, None, None]

                        if results_dict[param_name][i] is None:
                            # Initialise a list
                            results_dict[param_name][i] = []

                        results_dict[param_name][i].append(result_entry)
        return results_dict

    def get_best_vals(self, results_dicts, priors, fit_ld=True, return_combined=True):
        '''
        Gets the best values for a set of runs from the given results dicts

        Parameters
        ----------
        results_dicts : array_like, shape (n_batches, )
            The results_dicts obtained from get_results_dict
        fit_ld : bool, optional
            Should be True if LDC fitting. Default is True
        return_combined : bool, optional
            If True, will return the results dicts combined into a single dict.
            Default is True.

        Returns
        -------
        best_vals : dict
            Each entry is [best val, error]
        combined_dict : dict
            The combined results dictionary. Returned if return_combined is
            True
        '''
        print('Calculating best values for this run')
        best_vals = {}

        # Collate the results dicts
        combined_dict = self.combine_results_dicts(results_dicts)

        for param in combined_dict:
            # Loop through each parameter
            best_vals = self._initialise_dict_entry(best_vals, param)

            for i in np.ndindex(combined_dict[param].shape):
                if combined_dict[param][i] is not None:
                    # get the weighted avrage and error
                    if np.any(combined_dict[param][i][:,-1] == None):
                        # This is a constant
                        best_vals[param][i] = (combined_dict[param][i][:,0][0], None)
                    else:
                        best_vals[param][i] = weighted_avg_and_std(combined_dict[param][i][:, 0], combined_dict[param][i][:, -1], single_val=True)

        # Limb darkening bits
        if fit_ld:
            best_vals, combined_dict = self.add_best_u(best_vals, combined_dict)
        if return_combined:
            return best_vals, combined_dict
        return best_vals

    def combine_results_dicts(self, results_dicts):
        '''
        Combines the given results dicts into one dict

        Parameters
        ----------
        results_dicts : array_like, shape (n_batches, )
            The results_dicts obtained from get_results_dict

        Returns
        -------
        combined_dict : dict
            The combined results dictionary. Each entry is a list of values
            from the results dictionaries -
            [[best value, median, 16th percentile, 84th percentile, stdev],...]
        '''
        print('Combining results dicts')
        combined_dict = {}
        # Loop through each dict and the params
        for rd in results_dicts:
            for param in rd.keys():
                combined_dict = self._initialise_dict_entry(combined_dict, param)

                for i in np.ndindex(combined_dict[param].shape):
                    if rd[param][i] is not None:
                        if combined_dict[param][i] is None:
                            combined_dict[param][i] = rd[param][i]
                        else:
                            combined_dict[param][i] += rd[param][i]

        # Convert to np.arrays
        for param in combined_dict.keys():
            for i in np.ndindex(combined_dict[param].shape):
                if combined_dict[param][i] is not None:
                    combined_dict[param][i] = np.array(combined_dict[param][i])

        return combined_dict

    def add_best_u(self, best_dict, combined_dict):
        '''
        Given results dicts, adds in the u vals

        Parameters
        ----------
        best_dict : dict
            The dictionary of best results
        combined_dict : dict
            The full results

        Returns
        -------
        best_dict : dict
            The same dictionary with the best u vals added.
        combined_dict : dict
            The same dictionary with the u vals added.
        '''
        # Initialise each of the u params
        u_coeffs = []
        for param in self.ld_coeffs:
            ldc_q = 'q{}'.format(param[-1])
            ldc_u = 'u{}'.format(param[-1])
            u_coeffs.append(ldc_u)

            if ldc_u not in best_dict:
                combined_dict[ldc_u] = combined_dict[ldc_q].generate_blank_ParamArray()
                best_dict[ldc_u] = combined_dict[ldc_u].generate_blank_ParamArray()

        # Now get the values!
        for i in np.ndindex(best_dict['q0'].shape):
            if best_dict['q0'][i] is not None:
                # There are q values associated with this filter
                for u in u_coeffs:
                    best_dict[u][i] = []
                    combined_dict[u][i] = []

                # Put all the q values for a given filter into one place so we
                # can access q0, q1 simultaneously for converting to u
                n_batches = len(combined_dict['q0'][i])

                # indexing is filter_q[batch, val/err, qX]
                filter_q = np.zeros((n_batches, 2, len(self.ld_coeffs)))

                for b in range(n_batches):
                    for qi, q in enumerate(self.ld_coeffs):
                        filter_q[b,0,qi] = combined_dict[q][i][b][0]
                        filter_q[b,1,qi] = combined_dict[q][i][b][-1]

                # indexing is best_filter_q[val/err, qX]
                best_filter_q = np.vstack((best_dict[q][i] for q in self.ld_coeffs)).T

                # Convert the q values. First up the combined dict:
                for b in range(n_batches):
                    u, u_err = self.full_prior.ld_handler.convert_qtou_with_errors(*filter_q[b])

                    for k, uk in enumerate(u_coeffs):
                        combined_dict[uk][i].append([u[k], u_err[k]])

                # Now the best dict:
                u, u_err = self.full_prior.ld_handler.convert_qtou_with_errors(*best_filter_q)
                for k, uk in enumerate(u_coeffs):
                    best_dict[uk][i] = np.array([u[k], u_err[k]])

        return best_dict, combined_dict

    def _initialise_best_model(self, mode, global_prior, output_folder,
                               summary_file):
        '''
        Sets up the best fit model used to output best fit

        Parameters
        ----------
        mode : str
            The fitting mode being used
        global_prior : PriorInfo
            The full_prior from the retriever. Used to determine array shapes.

        Returns
        -------
        best_model : dict
            A dictionary containing param arrays of the best values

        Notes
        -----
        To do this, we go through all output parameter files and pull in the
        values. In folded mode, we pull in from the global summary first and
        add from the filter summaries after
        '''
        print('Initialising best fit model')
        mode = mode.lower()
        if mode not in ['all', 'batched', 'folded', '2_stage']:
            raise ValueError('Unrecognised mode {}'.format(mode))

        best_model_dict = {}

        # First we initialise each entry in the dict
        for param in global_prior.priors.keys():
            # Initialise from the global prior
            best_model_dict = self._initialise_dict_entry(best_model_dict, param, global_prior)

        for param in self.ld_coeffs:
            # Now initialise the LDC u params
            ldc_q = 'q{}'.format(param[-1])
            ldc_u = 'u{}'.format(param[-1])

            if ldc_u not in best_model_dict:
                best_model_dict[ldc_u] = best_model_dict[ldc_q].generate_blank_ParamArray()

        # Now we go through each of the output files and use them to populate
        # the best_model_dict

        # First use the top-level output
        top_output = pd.read_csv(os.path.join(output_folder, summary_file))

        # List of parameters which were fitted from the top level
        top_params = []

        for i, row in top_output.iterrows():
            param, tidx, fidx, eidx, best, err = row

            if param[-3:] == '/r*':
                param = param[:-3]
            if not err == '-' or mode in ['all', 'batched']:
                if tidx == '-':
                    tidx == None
                else:
                    tidx = int(tidx)

                if fidx == '-':
                    fidx == None
                else:
                    fidx = int(fidx)

                if eidx == '-':
                    eidx == None
                else:
                    eidx = int(eidx)

                best = float(best)
                if err == '-':
                    err = None
                else:
                    err = float(err)

                if param in best_model_dict:
                    if best_model_dict[param][tidx, fidx, eidx] is None:
                        best_model_dict[param][tidx, fidx, eidx] = []

                    best_model_dict[param][tidx, fidx, eidx].append([best, err])

                    if param not in top_params:
                        top_params.append(param)
                #if param in best_model_dict:
                #    best_model_dict[param][tidx, fidx, eidx] = [best, err]

        #if mode in ['all', 'batched']:
        #    self.best_model = best_model_dict
        #    return best_model_dict

        if mode == 'folded':
            # Now we have to go through the results for each of the filters and
            # add in the results from those
            for fi in range(global_prior.n_filters):
                path = os.path.join(output_folder, 'filter_{}_parameters'.format(fi), 'filter_{}_summary.csv'.format(fi))

                filter_output = pd.read_csv(path)
                for i, row in filter_output.iterrows():
                    param, tidx, fidx, eidx, best, err = row

                    if param[-3:] == '/r*':
                        param = param[:-3]

                    if tidx == '-':
                        tidx == None
                    else:
                        tidx = int(tidx)

                    if fidx == '-':
                        fidx == None
                    else:
                        fidx = int(fidx)

                    if eidx == '-':
                        eidx == None
                    else:
                        eidx = int(eidx)

                    best = float(best)
                    if err == '-':
                        err = None
                    else:
                        err = float(err)

                    if param in best_model_dict and param not in top_params:
                        # Store the value(s) from each if the parameter was not
                        # fitted in the folded run
                        if best_model_dict[param][tidx, fidx, eidx] is None:
                            best_model_dict[param][tidx, fidx, eidx] = []

                        best_model_dict[param][tidx, fidx, eidx].append([best, err])


        # Now we go through the params, checking to see if there are multiple
        # values. If there are, we need to take the weighted final values
        for param in best_model_dict:
            for i in np.ndindex(best_model_dict[param].shape):
                if best_model_dict[param][i] is not None:
                    param_results = np.array(best_model_dict[param][i])

                    if param_results[0,1] is None:
                        # deal with the global fixed values (err=None)
                        best_model_dict[param][i] = [param_results[0,0], None]
                        break
                    else:
                        best_model_dict[param][i] = weighted_avg_and_std(param_results[:, 0], param_results[:,1], single_val=True)

        self.best_model = best_model_dict

        return best_model_dict

    def _quicksave_result(self, results, priors, lightcurves,
                          base_output_path='./outputs', filter=None, batch=None):
        '''
        Quickly saves a batch result to file and pickle the Result and Prior
        objects
        '''
        result_dict = self.get_results_dict(results, priors, lightcurves)
        result_dict, _ = self.get_best_vals([result_dict], priors.fit_ld)
        base_fname = ''
        if filter is not None:
            base_fname += f'filter_{filter}_'
        else:
            base_fname += 'all_filters_'
        if batch is not None:
            base_fname += f'batch_{batch}_'
        result_file_fname = base_fname + 'output.csv'
        result_pickle_fname = base_fname +'results.pkl'
        priors_pickle_fname = base_fname + 'priors.pkl'

        # Quicksave best results
        output_path = os.path.join(base_output_path, 'quicksaves', result_file_fname)
        print(f'Quicksaving best results to {output_path}')
        self._save_results_dict(result_dict, output_path, False)

        # Quicksave full results object
        output_path = os.path.join(base_output_path, 'quicksaves', result_pickle_fname)
        print(f'Quicksaving full results to {output_path}')
        with open(output_path, 'wb') as f:
            try:
                pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Exception encountered:', e)

        # Quicksave priors
        output_path = os.path.join(base_output_path, 'quicksaves', priors_pickle_fname)
        print(f'Quicksaving priors to {output_path}')
        with open(output_path, 'wb') as f:
            try:
                pickle.dump(priors, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Exception encountered:', e)

    def _save_results_dict(self, results_dict, path, batched):
        '''
        Saves a dict to csv
        '''
        df = self._results_dict_to_dataframe(results_dict, batched)

        # Save outputs
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, na_rep='-')

    def _initialise_batman(self, all_lightcurves):
        '''
        Sets up batman so we can generate best-fit models for outputs

        Parameters
        ----------
        all_lightcurves : array_like
            Array containing the global set of light curves
        '''
        if self.best_model is None:
            raise ValueError('best-fit model is not intiialised.')

        if self.batman_initialised:
            return

        # Check that best model initialisation worked
        failed_key = []
        failed_index = []
        for key in self.best_model.keys():
            for i, lc in np.ndenumerate(all_lightcurves):
                if self.best_model[key][i] is None and lc is not None:
                    # If the failed key is a detrending coeff, we need to do
                    # some extra checking:
                    if key[0] == 'd':
                        method_idx = self.full_prior._detrend_method_index_array[i]
                        if key in self.full_prior.detrending_coeffs[method_idx]:
                            # If the key is associated with the lc, then this
                            # has failed.
                            failed_key.append(key)
                            failed_index.append(i)
                    else:
                        failed_key.append(key)
                        failed_index.append(i)
        if len(failed_key) > 0:
            print('Best model failed keys:', failed_key)
            print('Best model failed indices:', failed_index)
            raise RuntimeError('Something has gone wrong with the best model generation')


        print('Initialising best batman models')

        n_telescopes = all_lightcurves.shape[0]
        n_filters = all_lightcurves.shape[1]
        n_epochs = all_lightcurves.shape[2]

        # Set up a param array with the best values in
        self.batman_params = ParamArray('batman_params', (n_telescopes, n_filters, n_epochs), True, True, True, lightcurves=all_lightcurves)
        self.batman_models = ParamArray('batman_models', (n_telescopes, n_filters, n_epochs), True, True, True, lightcurves=all_lightcurves)

        for i in np.ndindex(all_lightcurves.shape):
            if all_lightcurves[i] is not None:

                # Set up the parameters
                transit_params = batman.TransitParams()
                transit_params.t0 = self.best_model['t0'][i][0]
                transit_params.per = self.best_model['P'][i][0]
                transit_params.rp = self.best_model['rp'][i][0]
                transit_params.a = self.best_model['a'][i][0]
                transit_params.inc = self.best_model['inc'][i][0]
                transit_params.ecc = self.best_model['ecc'][i][0]
                transit_params.w = self.best_model['w'][i][0]
                transit_params.u = [self.best_model[uX][i][0] for uX in self.ld_coeffs_u]
                transit_params.limb_dark = self.ld_model

                # Save the parameters
                self.batman_params[i] = transit_params

                # Make the model
                self.batman_models[i] = batman.TransitModel(transit_params, all_lightcurves[i].times)

        self.batman_initialised = True

    def _initialise_dict_entry(self, d, param, prior=None):
        '''
        Initialises param in results dictionaries using ParamArray

        If prior is not provided, defaults to self.full_prior
        '''
        if prior is None:
            prior = self.full_prior

        if param not in d:
            d[param] = prior.priors[param].generate_blank_ParamArray()
        return d

    def _batch_to_full_idx(self, i, param_name, lightcurves, allow_ttv):
        '''
        Converts a batch index into a full index

        Parameter
        ---------
        i : tuple
            (batch_tidx, batch_fidx, batch_eidx)

        Returns
        -------
        (tidx, fidx, eidx)
        '''
        batch_tidx, batch_fidx, batch_eidx = i

        if batch_tidx is None:
            tidx = None
        else:
            for k in np.ndindex(lightcurves.shape):
                if k[0] == batch_tidx and lightcurves[k] is not None:
                    tidx = lightcurves[k].telescope_idx

        if batch_fidx is None:
            fidx = None
        else:
            for k in np.ndindex(lightcurves.shape):
                if k[1] == batch_fidx and lightcurves[k] is not None:
                    fidx = lightcurves[k].filter_idx

        if batch_eidx is None:
            eidx = None
        else:
            for k in np.ndindex(lightcurves.shape):
                if k[2] == batch_eidx and lightcurves[k] is not None:
                    eidx = lightcurves[k].epoch_idx

        return tidx, fidx, eidx

    def _results_dict_to_dataframe(self, results_dict, batched):
        '''
        Converts a results dict to a pandas dataframe
        '''
        if batched:
            vals_arr = np.zeros((1, 7), dtype=object)
            columns = ['Parameter', 'Telescope', 'Filter', 'Epoch', 'Batch', 'Best', 'Error']
        else:
            vals_arr = np.zeros((1, 6), dtype=object)
            columns = ['Parameter', 'Telescope', 'Filter', 'Epoch', 'Best', 'Error']

        for param in results_dict:
            # Sort out display of the parameters
            if param == 'rp':
                print_param = 'rp/r*'
            elif param == 'a':
                print_param = 'a/r*'
            else:
                print_param = param

            for i in np.ndindex(results_dict[param].shape):

                if results_dict[param][i] is not None:

                    tidx, fidx, eidx = None, None, None
                    if results_dict[param].telescope_dependent:
                        tidx = i[0]
                    if results_dict[param].filter_dependent:
                        fidx = i[1]
                    if results_dict[param].epoch_dependent:
                        eidx = i[2]

                    if not batched:
                        # Add the best values in
                        if isinstance(results_dict[param][i][0], Iterable):
                            # DIRTY HACK due to weird behaviour with
                            # results from 'all' mode. I'm sorry, but I'm
                            # writing a thesis and don't have the time to
                            # find the actual cause.
                            vals_arr = np.append(vals_arr, np.array([[print_param, tidx, fidx, eidx, results_dict[param][i][0][0], results_dict[param][i][0][-1]]]), axis=0)
                        else:
                            vals_arr = np.append(vals_arr, np.array([[print_param, tidx, fidx, eidx, results_dict[param][i][0], results_dict[param][i][-1]]]), axis=0)

                        if param == 'a' and self.host_r is not None:
                            # Put a into AU as well
                            if isinstance(results_dict[param][i][0], Iterable):
                                # DIRTY HACK due to weird behaviour with
                                # results from 'all' mode
                                a_AU, a_AU_err = host_radii_to_AU(results_dict[param][i][0][0],
                                                                  self.host_r[0],
                                                                  results_dict[param][i][0][1],
                                                                  self.host_r[1], True)
                            else:
                                a_AU, a_AU_err = host_radii_to_AU(results_dict[param][i][0],
                                                                  self.host_r[0],
                                                                  results_dict[param][i][1],
                                                                  self.host_r[1], True)
                            vals_arr = np.append(vals_arr, np.array([['a/AU', tidx, fidx, eidx, a_AU, a_AU_err]]), axis=0)
                    else:
                        # Loop over batches
                        for bi in range(len(results_dict[param][i])):
                            vals_arr = np.append(vals_arr, np.array([[print_param, tidx, fidx, eidx, bi, results_dict[param][i][bi][0], results_dict[param][i][bi][-1]]]), axis=0)

                            if param == 'a' and self.host_r is not None:
                                # Put a into AU as well
                                a_AU, a_AU_err = host_radii_to_AU(results_dict[param][i][bi][0],
                                                                  self.host_r[0],
                                                                  results_dict[param][i][bi][1],
                                                                  self.host_r[1], True)
                                vals_arr = np.append(vals_arr, np.array([['a/AU', tidx, fidx, eidx, bi, a_AU, a_AU_err]]), axis=0)

        # Make the DataFrame - cut off the first (all zeros) entries
        return pd.DataFrame(vals_arr[1:], columns=columns)

    def _print_results(self, result):
        '''
        Prints a results dict to terminal
        '''
        df = self._results_dict_to_dataframe(result, False)

        print(df)

    ###########################################################################
    #               PLOTTING THINGS                                           #
    ###########################################################################
    def _plot_samples(self, result, prior, fname, folder='./plots'):
        '''
        Makes a corner plot of the samples from a result
        '''
        samples = result.samples
        best = result.best
        ndim = len(best)
        labels = prior.fitting_params[:,0]

        fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 12})


        # Add in the best value plots
        # Extract the axes
        axes = np.array(fig.axes).reshape((ndim, ndim))
        # Loop over the diagonal
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(best[i], color="g")

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(best[xi], color="g")
                ax.axhline(best[yi], color="g")
                ax.plot(best[xi], best[yi], "sg")

        fig.tight_layout()

        path = os.path.join(folder, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        fig.savefig(os.path.join(folder, fname), bbox_inches='tight', dpi=100)
        plt.close()

    def _plot_data(self, phase, flux, flux_err, model_phase, model_curve,
                       residuals, fname, title=None, folder='./plots',
                       figsize=(12,8), marker_color='dimgrey', line_color='black',
                       bin_data=False, cadence=2, binned_colour='red'):
            '''
            Plots the lightcurve and model consistently

            Parameters
            ----------

            cadence : float
                The cadence to bin to in phase, *not* minutes
            '''
            # Sort into phase orders
            data_order = np.argsort(phase)
            model_order = np.argsort(model_phase)

            phase = phase[data_order]
            flux = flux[data_order]
            if flux_err is not None:
                flux_err = flux_err[data_order]
            residuals = residuals[data_order]
            model_phase = model_phase[model_order]
            model_curve = model_curve[model_order]

            residuals_std = np.std(residuals)

            if bin_data:
                # Now we have to work out the binned values
                if flux_err is None:
                    err_for_bin = np.ones(len(flux))
                else:
                    err_for_bin = flux_err
                final_filter_lc = LightCurve(phase, flux, err_for_bin)
                binned_phase, binned_flux, binned_err, binned_residuals = final_filter_lc.bin(cadence, residuals)

                binned_residuals_std = np.std(binned_residuals)

                if flux_err is None:
                    binned_err = None

            # Set up the figure and the relevant axes
            gs = gridspec.GridSpec(6, 7)
            fig = plt.figure(figsize=figsize)

            main_ax = fig.add_subplot(gs[:-2, :-1])
            residual_ax = fig.add_subplot(gs[-2:, :-1], sharex=main_ax)
            hist_ax = fig.add_subplot(gs[-2:,-1], sharey=residual_ax)

            ####### PLOT! #########
            # Main plot
            main_ax.errorbar(phase, flux, flux_err,
                             zorder=1, capsize=2,
                             linestyle='', marker='.', color=marker_color,
                             elinewidth=0.8, alpha=0.6)

            main_ax.plot(model_phase, model_curve, zorder=3,
                         linewidth=2, color=line_color)
            if bin_data:
                main_ax.errorbar(binned_phase, binned_flux, binned_err,
                                 zorder=2, capsize=2, markersize=4,
                                 linestyle='', marker='.', color=binned_colour,
                                 elinewidth=0.8, alpha=0.6)

            # Residuals
            residual_ax.errorbar(phase, residuals, flux_err, zorder=1,
                                 linestyle='', marker='.', capsize=2,
                                 color=marker_color, elinewidth=0.8,
                                 alpha=0.6)

            residual_ax.axhline(0, linestyle='dashed', color='gray',
                                linewidth=1, zorder=3)

            if bin_data:
                residual_ax.errorbar(binned_phase, binned_residuals,
                                     binned_err, zorder=2, markersize=4,
                                     linestyle='', marker='.', capsize=2,
                                     color=binned_colour, elinewidth=0.8,
                                     alpha=0.6)

            # Histogram residuals
            rgba_color = colors.to_rgba(marker_color)
            facecolor = (rgba_color[0], rgba_color[1], rgba_color[2], 0.6)

            unbinned_counts, bins = np.histogram(residuals, bins=30)

            hist_ax.hist(bins[:-1], bins, weights=unbinned_counts,
                         orientation='horizontal', color=facecolor,
                         edgecolor=rgba_color, histtype='stepfilled')
            hist_ax.axhline(0, linestyle='dashed', color='gray',
                            linewidth=1, zorder=1)


            if bin_data:
                rgba_color = colors.to_rgba(binned_colour)
                facecolor = (rgba_color[0], rgba_color[1], rgba_color[2], 0.6)

                binned_counts, _ = np.histogram(binned_residuals, bins)

                weighted_binned_counts = binned_counts * unbinned_counts.max()/binned_counts.max()

                hist_ax.hist(bins[:-1], bins, weights=weighted_binned_counts,
                             orientation='horizontal', color=facecolor,
                             edgecolor=rgba_color, histtype='stepfilled',alpha=0.5)

                hist_ax.text(0.02, 0.93, r'$\sigma_{unbinned} = $' + str(round(residuals_std, 5)), transform=hist_ax.transAxes)
                hist_ax.text(0.02, 0.85, r'$\sigma_{binned}$ = ' + str(round(binned_residuals_std, 5)), transform=hist_ax.transAxes)


            # FORMATTING AXES
            # Prune axes
            main_ax.yaxis.set_major_locator(MaxNLocator(6, prune='lower'))
            residual_ax.yaxis.set_major_locator(MaxNLocator(4, prune='upper'))
            residual_ax.xaxis.set_major_locator(MaxNLocator(8, prune='upper'))

            # Add labels
            main_ax.set_ylabel('Normalised flux')
            residual_ax.set_ylabel('Residual')
            residual_ax.set_xlabel('Phase')

            # Format the axes
            main_ax.tick_params('both', which='both', direction='in',
                                labelbottom=False, top=True, right=True)

            residual_ax.tick_params('both', which='both', direction='in',
                                    top=True, right=True)

            hist_ax.tick_params('both', which='both', direction='in',
                                 labelleft=False, labelbottom=False,
                                 right=True, top=True)

            if title is not None:
                main_ax.set_title(title)

            fig.tight_layout()
            fig.subplots_adjust(hspace=0, wspace=0)

            os.makedirs(os.path.dirname(os.path.join(folder, fname)), exist_ok=True)

            # SAVE THE PLOT
            fig.savefig(os.path.join(folder, fname), bbox_inches='tight', dpi=300)

            plt.close()
