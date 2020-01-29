'''
plotting.py


Plotting module for TransitFit
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import to_rgb
from matplotlib import colors
import batman
import os

def plot_individual_lightcurves(times, flux, uncertainty, priorinfo, results,
                                folder_path='./plots', figsize=(12,8),
                                color='dimgrey', titles=None, add_titles=True,
                                fnames=None):
    '''
    Once you have run retrieval, use this to plot things!

    Will make a figure for each light curve, and save with filenames reflecting
    epoch and filter number
    '''
    # Get some numbers for loop purposes
    n_epochs = priorinfo.num_times
    n_filters = priorinfo.num_wavelengths

    if titles is not None:
        try:
            titles = np.array(titles)
            if not titles.shape == (n_filters, n_epochs):
                print('Warning: titles shape is {}, not ({},{}). Reverting to default titles'.format(titles.shape, n_filters, n_epochs))
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

    # For each light curve, make a plot!
    for fi in range(n_filters):
        for ei in range(n_epochs):
            if times[fi, ei] is not None:
                # We are plotting this!!

                # Set up the figure and the relevant axes
                gs = gridspec.GridSpec(6, 7)
                fig = plt.figure(figsize=figsize)

                main_ax = fig.add_subplot(gs[:-2, :-1])
                residual_ax = fig.add_subplot(gs[-2:, :-1], sharex=main_ax)
                hist_ax = fig.add_subplot(gs[-2:,-1], sharey=residual_ax)

                # Format the axes
                main_ax.tick_params('both', which='both', direction='in',
                                    labelbottom='off', top='on', right='on')


                residual_ax.tick_params('both', which='both', direction='in',
                                        top='on', right='on')


                hist_ax.tick_params('both', which='both', direction='in',
                                     labelleft='off', labelbottom='off',
                                     right='on', top='on')

                main_ax.set_ylabel('Normalised flux')
                residual_ax.set_ylabel('Residual')
                residual_ax.set_xlabel('Time (BJD)')

                if add_titles:
                    if titles is None:
                        main_ax.set_title('Filter {}, Epoch {}'.format(fi, ei))
                    else:
                        main_ax.set_title(titles[fi, ei])

                fig.tight_layout()
                fig.subplots_adjust(hspace=0, wspace=0)


                # Now make the best fit light curve

                # First we set up the parameters
                params = batman.TransitParams()
                params.t0 = best_dict['t0']
                params.per = best_dict['P']
                params.rp = best_dict['rp'][fi]
                params.a = best_dict['a']
                params.inc = best_dict['inc']
                params.ecc = best_dict['ecc']
                params.w = best_dict['w']
                params.limb_dark = priorinfo.limb_dark
                params.u = np.array([best_dict[key] for key in priorinfo.limb_dark_coeffs]).T[fi]

                plot_times = np.linspace(times[fi, ei].min(), times[fi,ei].max(), 1000 )

                m = batman.TransitModel(params, plot_times)

                best_curve = m.light_curve(params)

                m_sample_times = batman.TransitModel(params, times[fi, ei])
                time_wise_best_curve = m_sample_times.light_curve(params)

                norm = best_dict['norm'][fi, ei]

                # Plot the raw data
                if priorinfo.detrend:
                    d = [best_dict[d][fi, ei] for d in priorinfo.detrending_coeffs]

                    dF = priorinfo.detrending_function(times[fi, ei]-np.floor(times[fi, ei][0]), *d)

                    plot_fluxes = norm * (flux[fi, ei] - dF)
                else:
                    plot_fluxes = norm * (flux[fi, ei])

                main_ax.errorbar(times[fi, ei], plot_fluxes,
                            norm * uncertainty[fi, ei], zorder=1,
                            linestyle='', marker='x', color=color,
                            elinewidth=0.8, alpha=0.6)

                # Plot the curve
                main_ax.plot(plot_times, best_curve, linewidth=2,
                            color=color)

                # plot the residuals
                residuals = plot_fluxes - time_wise_best_curve

                residual_ax.errorbar(times[fi, ei], residuals,
                            norm * uncertainty[fi, ei], linestyle='',
                            color=color, marker='x', elinewidth=0.8, alpha=0.6)

                residual_ax.axhline(0, linestyle='dashed', color='gray',
                                    linewidth=1, zorder=1)

                # Histogram the residuals
                # Sort out colors:
                rgba_color = colors.to_rgba(color)
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

                # Save the figures
                # Make the plots folder
                os.makedirs(folder_path, exist_ok=True)

                if fnames is None:
                    fig.savefig('{}/f{}_e{}.pdf'.format(folder_path, fi, ei),
                                bbox_inches='tight')
                else:
                    if fnames[fi,ei] is None:
                        fig.savefig('{}/f{}_e{}.pdf'.format(folder_path, fi, ei),
                                    bbox_inches='tight')
                    else:
                        if not fnames[fi,ei][-4:] == '.pdf':
                            fnames[fi,ei] += '.pdf'
                        fig.savefig(os.path.join(folder_path, fnames[fi,ei]),
                                    bbox_inches='tight')
