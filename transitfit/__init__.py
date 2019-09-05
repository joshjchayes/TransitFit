'''
TransitFit package

This package is designed to fit transit light curves using BATMAN
'''
name = 'transitfit'


from .retriever import Retriever
from .priorinfo import PriorInfo, setup_priors
from .filereader import read_data_file, read_priors_file
