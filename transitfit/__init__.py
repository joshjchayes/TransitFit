'''
TransitFit package

This package is designed to fit transit light curves using BATMAN
'''
name = 'transitfit'


from .retriever import Retriever
from .priorinfo import PriorInfo, setup_priors
from .io import read_data_file, read_priors_file, read_data_file_array
from ._pipeline import run_retrieval_from_paths
