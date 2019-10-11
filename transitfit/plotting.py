'''
plotting.py


Plotting module for TransitFit
'''

import numpy as np
import matplotlib.pyplot as plt
import batman
import os

def plot_best(times, flux, uncertainty, priorinfo, results, input_file=None,
              **subplots_kwargs):
    '''
    Once you have run retrieval, use this to plot things!

    Will make a figure for each light curve, and save with filenames either
    with epoch and filter number, or reflecting the original filenames of the
    data if input_file is given and points to a file which can be read by
    transitfit.io.read_input_file
    '''
    # Get some numbers for loop purposes
    n_epochs = priorinfo.num_times
    n_filters = priorinfo.num_wavelengths

    # Get the best values
    best = results.samples[np.argmax(results.logl)]

    best_dict = priorinfo._interpret_param_array(best)

    # For each light curve, make a plot!
    for fi in range(n_filters):
        for ei in range(n_epochs):
            if times[fi, ei] is not None:
                # We are plotting this!!

                # Make the figure
                fig, ax = plt.subplots(**subplots_kwargs)


                # Now make the best fit light curve

                # First we set up the parameters
                params = batman.TransitParams()
                params.t0 = best_dict['t0'][ei]
                params.per = best_dict['P']
                params.rp = best_dict['rp'][fi]
                params.a = best_dict['a']
                params.inc = best_dict['inc']
                params.ecc = best_dict['ecc']
                params.w = best_dict['w']
                params.limb_dark = priorinfo.limb_dark
                params.u = np.array([best_dict[key] for key in priorinfo.limb_dark_coeffs]).T[fi]

                m = batman.TransitModel(params, times[fi, ei])


                best_curve = m.light_curve(params)

                norm = best_dict['norm'][fi, ei]
                shift = best_dict['shift'][fi, ei]
                print(norm)


                # Plot the raw data
                if priorinfo.detrend:
                    d = [best_dict[d][fi, ei] for d in priorinfo.detrending_coeffs]

                    print(d)

                    dF = priorinfo.detrending_function(times[fi, ei]-np.floor(times[fi, ei][0]), *d)

                    ax.errorbar(times[fi, ei], norm * (flux[fi, ei] + shift - dF),
                                norm * uncertainty[fi, ei], zorder=1, fmt='bx',
                                alpha=0.8)
                else:
                    ax.errorbar(times[fi, ei], norm * (flux[fi, ei] + shift),
                                norm * uncertainty[fi, ei], zorder=1, fmt='bx',
                                alpha=0.8)




                # Plot the curve
                ax.plot(times[fi,ei], best_curve, linewidth=2)

                # Add labels
                ax.set_xlabel('Time')
                ax.set_ylabel('Normalised flux')
                ax.set_title('Filter {}, Epoch {}'.format(fi, ei))

                # Save the figures
                # Make the plots folder
                os.makedirs('plots', exist_ok=True)
                fig.savefig('plots/f{}_e{}.pdf'.format(fi, ei))
