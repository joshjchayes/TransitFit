'''
Plotting module for TransitFit
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import to_rgb
from matplotlib import colors
import batman
import os

def plot_individual_lightcurves(lightcurves, priorinfo, results,
                                folder_path='./plots', figsize=(12,8),
                                marker_color='dimgrey', line_color='black',
                                titles=None, add_titles=True, fnames=None,
                                plot_phase=False, t0=None, period=None):
    '''
    Once you have run retrieval, use this to plot things!

    Will make a figure for each light curve, and save with filenames reflecting
    telescope, filter, and epoch number

    plot_phase : bool, optional
        If True and t0 and period are provided, will plot using phase rather
        than time on x axis. This is useful for folded curves.
    t0 : float, optional
        The value of t0 that the lightcurves are folded to. This is treated as
        phase 0.5 (centred in plot).
    period : float, optional
        The period that lightcurves are folded with.
    '''
    # Get some numbers for loop purposes
    n_epochs = priorinfo.n_epochs
    n_filters = priorinfo.n_filters

    if titles is not None:
        try:
            titles = np.array(titles)
            if not titles.shape == lightcurves.shape:
                print('Warning: titles shape is {}, not {}. Reverting to default titles'.format(titles.shape, lightcurves.shape))
                titles = None
        except Exception as e:
            print('Raised exception: {}'.format(e))
            print('Reverting to default titles')
            titles = None


    if fnames is not None:
        try:
            fnames = np.array(fnames)
            if not fnames.shape == (n_filters, n_epochs):
                print('Warning: fnames shape is {}, not ({},{}). Reverting to default fnames'.format(fnames.shape, n_filters, n_epochs))
                fnames = None
        except Exception as e:
            print('Raised exception: {}'.format(e))
            print('Reverting to default fnames')
            fnames = None

    # Get the best values
    best = results.samples[np.argmax(results.logl)]

    best_dict = priorinfo._interpret_param_array(best)

    # Get the array of detrending coeffs:
    if priorinfo.detrend:
        # We need to combine the detrending coeff arrays into one
        # Each entry should be a list containing all the detrending
        # coefficients to trial.
        d = np.full(lightcurves.shape, None, object)

        for i in np.ndindex(d.shape):
            for coeff in np.ravel(priorinfo.detrending_coeffs):
                if best_dict[coeff][i] is not None:
                    if d[i] is None:
                        d[i] = [best_dict[coeff][i]]
                    else:
                        d[i].append(best_dict[coeff][i])


    # For each light curve, make a plot!
    for i in np.ndindex(lightcurves.shape):
        if lightcurves[i] is not None:
            # We are plotting this!!
            telescope_idx = lightcurves[i].telescope_idx
            filter_idx = lightcurves[i].filter_idx
            epoch_idx = lightcurves[i].epoch_idx


            # Set up the figure and the relevant axes
            gs = gridspec.GridSpec(6, 7)
            fig = plt.figure(figsize=figsize)

            main_ax = fig.add_subplot(gs[:-2, :-1])
            residual_ax = fig.add_subplot(gs[-2:, :-1], sharex=main_ax)
            hist_ax = fig.add_subplot(gs[-2:,-1], sharey=residual_ax)

            # Work out x axis values and labels
            if not plot_phase:
                # x axis is time, not phase
                x_vals = lightcurves[i].times
                x_label = 'Time (BJD)'
            else:
                # x axis is time, need to calculate phase
                try:
                    x_vals = (lightcurves[i].times - (t0 + (period/2)))/period
                    x_label = 'Phase'
                except Exception as e:
                    print('Exception raised when calculating phase. Reverting to plotting time')
                    print(e)
                    x_vals = lightcurves[i].times
                    x_label= 'Time (BJD)'

            plot_times = np.linspace(lightcurves[i].times.min(), lightcurves[i].times.max(), 1000 )

            # Now make the best fit light curve

            # First we set up the parameters
            params = batman.TransitParams()
            params.t0 = best_dict['t0'][i]
            params.per = best_dict['P'][i]
            params.rp = best_dict['rp'][i]
            params.a = best_dict['a'][i]
            params.inc = best_dict['inc'][i]
            params.ecc = best_dict['ecc'][i]
            params.w = best_dict['w'][i]
            params.limb_dark = priorinfo.limb_dark

            if priorinfo.fit_ld:
                # NOTE needs converting from q to u
                best_q = np.array([best_dict[key][i] for key in priorinfo.limb_dark_coeffs])
            else:
                q = np.array([priorinfo.priors[key][i] for key in priorinfo.limb_dark_coeffs])
                for j in np.ndindex(q.shape):
                    q[j] = q[j].default_value
                best_q = q

            params.u = priorinfo.ld_handler.convert_qtou(*best_q)

            m = batman.TransitModel(params, plot_times)

            best_curve = m.light_curve(params)

            m_sample_times = batman.TransitModel(params, lightcurves[i].times)
            time_wise_best_curve = m_sample_times.light_curve(params)

            norm = best_dict['norm'][i]

            # Plot the raw data
            if priorinfo.detrend:
                plot_fluxes, plot_errors = lightcurves[i].detrend_flux(d[i], norm)
            else:
                plot_fluxes, plot_errors = lightcurves[i].detrend_flux(None, norm)

            main_ax.errorbar(x_vals, plot_fluxes,
                        plot_errors, zorder=1,
                        linestyle='', marker='x', color=marker_color,
                        elinewidth=0.8, alpha=0.6)

            # Plot the curve
            # Convert to phase if needed:
            if plot_phase:
                plot_times = (plot_times - t0 + (period/2))/period

            main_ax.plot(plot_times, best_curve, linewidth=2,
                        color=line_color)

            # plot the residuals
            residuals = plot_fluxes - time_wise_best_curve

            residual_ax.errorbar(x_vals, residuals,
                        plot_errors, linestyle='',
                        color=marker_color, marker='x', elinewidth=0.8, alpha=0.6)

            residual_ax.axhline(0, linestyle='dashed', color='gray',
                                linewidth=1, zorder=1)

            # Histogram the residuals
            # Sort out colors:
            rgba_color = colors.to_rgba(marker_color)
            facecolor = (rgba_color[0], rgba_color[1], rgba_color[2], 0.6)

            hist_ax.hist(residuals, bins=30, orientation='horizontal',
                         color=facecolor, edgecolor=rgba_color,
                         histtype='stepfilled')
            hist_ax.axhline(0, linestyle='dashed', color='gray',
                            linewidth=1, zorder=1)


            # Prune axes
            main_ax.yaxis.set_major_locator(MaxNLocator(6, prune='lower'))
            residual_ax.yaxis.set_major_locator(MaxNLocator(4, prune='upper'))
            residual_ax.xaxis.set_major_locator(MaxNLocator(8, prune='upper'))

            # Add labels
            main_ax.set_ylabel('Normalised flux')
            residual_ax.set_ylabel('Residual')
            residual_ax.set_xlabel(x_label)

            # Format the axes
            main_ax.tick_params('both', which='both', direction='in',
                                labelbottom=False, top=True, right=True)

            residual_ax.tick_params('both', which='both', direction='in',
                                    top=True, right=True)

            hist_ax.tick_params('both', which='both', direction='in',
                                 labelleft=False, labelbottom=False,
                                 right=True, top=True)


            if add_titles:
                if titles is None:
                    if plot_phase:
                        title = 'Phase plot for filter {}'.format(filter_idx)
                    else:
                        title = 'Fitted curve: Telescope {} Filter {}, Epoch {}'.format(telescope_idx, filter_idx, epoch_idx)
                else:
                    title = titles[i]

                main_ax.set_title(title)

            fig.tight_layout()
            fig.subplots_adjust(hspace=0, wspace=0)

            # Save the figures
            # Make the plots folder
            os.makedirs(folder_path, exist_ok=True)

            if fnames is None:
                if plot_phase:
                    fname = 'f{}_phase_plot.pdf'.format(filter_idx)
                else:
                    fname = 't{}_f{}_e{}.pdf'.format(telescope_idx, filter_idx, epoch_idx)
            else:
                if fnames[i] is None:
                    fname = 't{}_f{}_e{}.pdf'.format(telescope_idx, filter_idx, epoch_idx)
                else:
                    if not fnames[i][-4:] == '.pdf':
                        fnames[i] += '.pdf'
                    fname = fnames[i]

            fig.savefig(os.path.join(folder_path, fname), bbox_inches='tight')

            plt.close()


def quick_plot(lightcurve, fname, folder_path, t0=None, period=None):
    '''
    Quickly plots a single lightcurve on some axes

    If t0 and period are provided, will use a phase plot
    '''
    base_fname = fname

    if t0 is None or period is None:
        # Plot the raw times with t0 added
        fig, ax = plt.subplots()
        x_vals = lightcurve.times
        x_label= 'Time (BJD)'
        ax.errorbar(x_vals, lightcurve.flux, lightcurve.errors, zorder=1,
            linestyle='', marker='x', color='dimgrey', elinewidth=0.8, alpha=0.6)

        if t0 is not None:
            ax.axvline(t0, linestyle='dashed', color='gray',
                            linewidth=1, zorder=1)

        ax.set_xlabel(x_label)
        ax.set_ylabel('Normalised flux')

        ax.tick_params('both', which='both', direction='in',
                            labelbottom=True, top='on', right='on')

        os.makedirs(folder_path, exist_ok=True)

        if not base_fname[-4:] == '.png':
            fname = base_fname + '_raw_times.png'
        else:
            fname = base_fname[:-4] + '_raw_times.png'

        fig.savefig(os.path.join(folder_path, fname),
                    bbox_inches='tight')

        return

    # Plot with phases
    # Plot the raw times with t0 added
    fig, ax = plt.subplots()

    x_vals = lightcurve.get_phases(t0, period)
    x_label = 'Phase'

    ax.errorbar(x_vals, lightcurve.flux, lightcurve.errors, zorder=1,
        linestyle='', marker='x', color='dimgrey', elinewidth=0.8, alpha=0.6)

    ax.axvline(0.5, linestyle='dashed', color='gray',
               linewidth=1, zorder=1)

    ax.set_xlabel(x_label)
    ax.set_ylabel('Normalised flux')

    ax.tick_params('both', which='both', direction='in',
                        labelbottom=True, top='on', right='on')

    os.makedirs(folder_path, exist_ok=True)

    if not base_fname[-4:] == '.png':
        fname = base_fname + '_phase.png'
    else:
        fname = base_fname[:-4] + '_phase.png'

    fig.savefig(os.path.join(folder_path, fname),
                bbox_inches='tight')


def plot_from_file(path, phase_plot=True, folder_path='./plots',
                   figsize=(12,8), marker_color='dimgrey',
                   line_color='black', title=None, save_path=None):
    '''
    Plots a light curve from a file.

    If phase_plot is true, then will plor phase on x axis. Otherwise plots time.

    '''
    # Get the data
    time, phase, flux, flux_err, model = pd.read_csv(path).values.T
    residuals = flux - model

    # Set up the figure and the relevant axes
    gs = gridspec.GridSpec(6, 7)
    fig = plt.figure(figsize=figsize)

    main_ax = fig.add_subplot(gs[:-2, :-1])
    residual_ax = fig.add_subplot(gs[-2:, :-1], sharex=main_ax)
    hist_ax = fig.add_subplot(gs[-2:,-1], sharey=residual_ax)

    if phase_plot:
        x_vals = phase
        x_label = 'Phase'
    else:
        x_vals = time
        x_label = 'Time (BJD)'

    main_ax.errorbar(x_vals, flux,
                     flux_err, zorder=1,
                     linestyle='', marker='x', color=marker_color,
                     elinewidth=0.8, alpha=0.6)

    # Plot the curve
    main_ax.plot(xvals, model, linewidth=2,
                color=line_color)

    # plot the residuals
    residual_ax.errorbar(x_vals, residuals,
                flux_err, linestyle='',
                color=marker_color, marker='x', elinewidth=0.8, alpha=0.6)

    residual_ax.axhline(0, linestyle='dashed', color='gray',
                        linewidth=1, zorder=1)

    # Histogram the residuals
    # Sort out colors:
    rgba_color = colors.to_rgba(marker_color)
    facecolor = (rgba_color[0], rgba_color[1], rgba_color[2], 0.6)

    hist_ax.hist(residuals, bins=30, orientation='horizontal',
                 color=facecolor, edgecolor=rgba_color,
                 histtype='stepfilled')
    hist_ax.axhline(0, linestyle='dashed', color='gray',
                    linewidth=1, zorder=1)


    # Prune axes
    main_ax.yaxis.set_major_locator(MaxNLocator(6, prune='lower'))
    residual_ax.yaxis.set_major_locator(MaxNLocator(4, prune='upper'))
    residual_ax.xaxis.set_major_locator(MaxNLocator(8, prune='upper'))

    # Add labels
    main_ax.set_ylabel('Normalised flux')
    residual_ax.set_ylabel('Residual')
    residual_ax.set_xlabel(x_label)

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

    if save_path is None:
        save_path = os.path.splitext(path)[0] + '_plot.pdf'

    fig.savefig(save_path, bbox_inches='tight')

    plt.close()
