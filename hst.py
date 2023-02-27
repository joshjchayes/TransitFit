# Sample script to work with HST detrending.

import numpy as np
import copy
from transitfit import run_retrieval

# The start times of each visit ! ( for calculating t_visit )
t_v0 = np.array([2457459.914186])

# Set up the custom detrending function


def HST_detrending(lightcurve, s, v1, v2, a, b, t0, P):
    times = copy.deepcopy(lightcurve.times)
    flux = copy.deepcopy(lightcurve.flux)

    # Find t_orb ( see below for function definition )
    t_orb = get_t_orb(times)

    # Get t_visit
    t_visit_0 = t_v0[t_v0 < times[0]][-1]
    t_visit = times - t_visit_0

    # s ( t ) : we assume that the first exposure is reverse scanning
    # ( since first exposure of each orbit is dropped )
    S = np.empty(len(times))
    S[::2] = s
    S[1::2] = 1

    f_sys = (S + v1 * t_visit + v2 * (t_visit ** 2)) * \
        (1 - np . exp(- a * t_orb - b))

    return flux/f_sys


def get_t_orb(times):
    # Work out the observation cadence
    cadence = times[1] - times[0]

    # Find the orbit starts - is the preceding observation more than
    # 2 xcadence away ?
    diff = times[1:] - times[: -1]

    # How many orbits are in the observation ?
    n_orbits = np.sum(diff > 2 * cadence) + 1

    # +1 gives us the index of the first exposure in an orbit !
    orbit_start_indices = np.where(diff > 2 * cadence)[0] + 1

    orbit_bounds = np.concatenate(
        (np.array([0]), orbit_start_indices, np.array([len(times)])))

    # Calculate the time since first exposure in orbit
    # Since the first observation in the orbit is dropped , we need to add
    # 1 cadence to the times
    t_orb = np.concatenate((np.array([times[orbit_bounds[i]: orbit_bounds[i + 1]] -
                                      times[orbit_bounds[i]] for i in range(n_orbits)]))) + cadence

    return t_orb


# Set up the host info, using arbitrary values.
# These are all given in (value, uncertainty) tuples
host_T = (4732.171, 100)  # PySSED
host_z = (-0.0633, 0.1)  # GAIA DR3
host_logg = (4.4843, 0.1)  # GAIA DR3

# Set up the detrending models
detrending_models = [['nth order', 2],  # detrending index 0
                     ['custom', HST_detrending, [], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]  # detrending index 1

# Set the detrending coefficient bounds
detrending_limits = [[-10, 10], [-10, 10]]

# Now we can run the retrieval!
results = run_retrieval('input_data.csv', 'priors.csv', 'filter_profiles.csv',
                        detrending_list=detrending_models, check_batchsizes=False,
                        detrending_limits=detrending_limits,
                        ld_fit_method='coupled',
                        host_T=host_T, host_logg=host_logg, host_z=host_z, allow_ttv=True)
