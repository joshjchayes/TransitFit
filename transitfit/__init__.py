'''
TransitFit package

This package is designed to fit transit light curves using BATMAN
'''
name = 'transitfit'
__version__ = '2.3.1'

from .retriever import Retriever
from .priorinfo import PriorInfo
from .lightcurve import LightCurve
from ._pipeline import run_retrieval
from ._utils import split_lightcurve_file, calculate_logg, AU_to_host_radii
from .lightcurve import LightCurve
