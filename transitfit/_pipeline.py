'''
pipeline

A function which will run everything when given a path to light curve and
priors!
'''

from .filereader import read_data_file, read_priors_file
from .retriever import Retriever

def run_retrieval_from_paths(data_paths, prior_path):
    '''
    Runs a full retrieval when given data_paths and prior_path

    Parameters
    ----------
    data_paths : str or list of str
        Paths to the files which contain the light curve data. Can either be
        a single string path, or a list of paths as strings
    prior_path : str
        Path to the prior .csv file

    Returns
    -------
    results : dict
        The results returned by Retriever.run_dynesty()
    '''
    # Read in the data
    print('Loading light cuve data...')
    if type(data_paths) == str:
        data_paths = [data_paths]

    data = [read_data_file(path) for path in data_paths]

    times = [d[0] for d in data]
    depths = [d[1] for d in data]
    errors = [d[2] for d in data]

    # Read in the priors
    print('Loading priors from {}'.format(prior_path))
    priors = read_priors_file(prior_path)

    print('Beginning retrieval')
    retriever = Retriever()
    return retriever.run_dynesty(times, depths, errors, priors)
