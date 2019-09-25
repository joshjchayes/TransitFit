'''
pipeline

A function which will run everything when given a path to light curve and
priors!
'''

from .io import read_data_file_array, read_priors_file, read_input_file
from .retriever import Retriever
import numpy as np

def run_retrieval_from_paths(input_csv_path, prior_path, limb_dark='quadratic',
                             detrending=None, nlive=300):
    '''
    Runs a full retrieval when given data_paths and prior_path

    Parameters
    ----------
    inptu_csv_path : str
        Path to the file which contains the paths to data files with associated
        epoch and filter numbers. For info on format, see documentation for
        transitfit.io.read_input_file
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
    print('Loading light curve data...')
    times, depths, errors = read_input_file(input_csv_path)

    # Read in the priors
    print('Loading priors from {}'.format(prior_path))
    priors = read_priors_file(prior_path, limb_dark)

    if detrending is not None:
        print('Initialising detrending')
        priors.add_detrending(times, detrending)

    #print(times)
    #print(priors.priors)

    print('Beginning retrieval')
    retriever = Retriever()
    return retriever.run_dynesty(times, depths, errors, priors, nlive=nlive)
