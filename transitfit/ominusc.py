'''
ominusc

bits for O-C analysis

'''

import numpy as np
import matplotlib.pyplot as plt

def plot_OC(t0, P, base_t0=None):
    '''
    Makes an O - C plot for the given t0 values, using t0[0] as the base t0

    Parameters
    ----------
    t0 : array_like, shape (n_points, 2)
        The t0 values and their uncertainties. The first entry is used to
        calculate the predicted t0 times using the period unless base_t0 is
        provided.
    P : array_like, shape (2, )
        The period of the orbit and the uncertainty
    base_t0 : array_like, shape (2, ), optional
        If provided, will base the predicted t0 values off this, rather than
        the first entry in t0. Should be provided as value and uncertainty
    '''
    t0 = np.array(t0)

    if base_t0 is None:
        base_t0 = t0[0]

    # Set up array for the o-c values (denoted as d in code)
    d = np.zeros(t0.shape)
    epoch_n = np.zeros(len(t0))

    for i in range(len(d)):
        # Calculate the o-c value and uncertainty

        # Calculate which epoch we are in
        n_upper = base_t0[0] + np.ceil(t0[i,0]/P[0])
        n_lower = base_t0[0] + np.floor(t0[i,0]/P[0])

        d_upper = t0[i,0] - n_upper * P[0]
        d_lower = t0[i,0] - n_lower * P[0]

        if abs(d_upper) > abs(d_lower):
            d[i, 0] = d_lower
            epoch_n[i] = n_lower
        else:
            d[i, 0] = d_upper
            epoch_n[i] = n_lower

        # now calculate uncertainty
        d[i, 1] = d[i, 0] * np.sqrt((t0[i,1]/t0[i, 0])**2 + (base_t0[1]/base_t0[0])**2 + (P[1]/P[0])**2)

    # Now we can plot!
    fig, ax = plt.subplots()

    ax.errorbar(epoch_n, d[:,0], yerr=d[:, 1])

    # Add a dotted line at d = 0
    ax.axhline(0, linestyle='dashed', color='gray', linewidth=1, zorder=1)

    # Add labels
    ax.set_xlabel('Epochs since t0')
    ax.set_ylabel('O - C')
