'''
pipeline

A function which will run everything when given a path to light curve and
priors!
'''

from .io import read_data_file_array, read_priors_file
from .retriever import Retriever
import numpy as np

def run_retrieval_from_paths(data_paths, prior_path, limb_dark='quadratic',
                             detrending=None, nlive=300):
    '''
    Runs a full retrieval when given data_paths and prior_path

    Parameters
    ----------
    data_paths : str or list of str
        Paths to the files which contain the light curve data. Can either be
        a single string path, or a list of paths as strings
    prior_path : str
        Path to the prior .csv file
    limb_dark : str
        The limb darkening model to use. Default is quadratic
    detrending : str or None
        If not none, detrending will be performed. Accepted detrending models
        are 'linear', 'quadratic', 'sinusoidal'. Default is None

    Returns
    -------
    results : dict
        The results returned by Retriever.run_dynesty()
    '''
    data_paths = np.array(data_paths)

    if not data_paths.ndim == 2:
        raise ValueError('Data paths should have 2 dimensions. Rows are wavelengths and columns are epochs.')

    # Read in the data
    print('Loading light curve data...')
    times, depths, errors = read_data_file_array(data_paths)

    # Read in the priors
    print('Loading priors from {}'.format(prior_path))
    priors = read_priors_file(prior_path, limb_dark)

    print(detrending)
    if detrending is not None:
        print('Initialising detrending')
        priors.add_detrending(times, detrending)

        print(priors.detrending_coeffs)
        print(priors.priors)


    #print(times)
    #print(priors.priors)

    print('Beginning retrieval')
    retriever = Retriever()
    return retriever.run_dynesty(times, depths, errors, priors, nlive=nlive)
