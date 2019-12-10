
# TransitFit

**Author** - Joshua Hayes (joshjchayes@gmail.com)

TransitFit is a package designed to run retrieval on exoplanet transit light curves. It offers the functionality to simultaneously fit multi-epoch and multi-wavelength light curves, along with the functionality to couple host limb-darkening parameters to phhysical models across different wavelengths (through [Limb darkening toolkit (ldtk)](https://github.com/hpparvi/ldtk)). Currently, the retrieval algorithm used is nested sampling through [dynesty](https://dynesty.readthedocs.io/en/latest/index.html), and the transit light curve modelling is handled by [batman](https://www.cfa.harvard.edu/~lkreidberg/batman/index.html)

## Contents

[Requirements](#requirements)

[Overview](#overview)

[Input files](#inputs)

[Limb darkening](#limb_darkening)

[Detrending](#detrending)

<a name="requirements"></a>
## Requirements
Along with an installation of Python 3 (with the standard Conda distribution packages), TransitFit requires the following packages to be installed:

- [dynesty](https://dynesty.readthedocs.io/en/latest/index.html)

- [batman](https://www.cfa.harvard.edu/~lkreidberg/batman/index.html)

- [Limb darkening toolkit (ldtk)](https://github.com/hpparvi/ldtk)

<a name="overview"></a>
## Overview
TransitFit can be used in a number of ways, but the easiest is to use the ``run_retrieval_from_paths`` function. This reads in a series of user-defined input files and uses them along with supplied arguments to run a retrieval directly. This requires three files to be provided, which provide TransitFit with paths to the data, information on the filters observations were taken with, and prior values for parameters to be fitted. The format for each of these files is discussed below. 

In order to handle data from different epochs and filters, TransitFit uses a grid-like system for dealing with light curves. Each row in the grid is a filter, whilst each column is a different epoch. By assigning an index to each different filter and epoch, TransitFit makes it easy to fit spectrographic data, where multiple filters are taken simultaneously, photometric data, where single filter light curves are taken in an epoch, or a combination of the two. 

TransitFit can fit for physical parameters, including orbital period, planet radius, and limb darkening parameters. Alongside, it is also possible to fit for systematics and perform detrending simulateously with fitting of physical parameters. The detrending can be either through a polynomial of user-specified order, or a user-supplied function. More information can be found [here](#detrending).

<a name="inputs"></a>
## Input files
There are three input files which are required by TransitFit, along with the light curve data themselves. These are the data path file, the priors file, and (if utilising the functionality to couple limb darkeing across wavelengths) the filter info file. The layout of these files is strict and care should be taken when creating them. All three of these files include a header row. 

### Data path file
This file should contain 3 columns: path, epoch, and filter. The path entry is the path to a file contining data for a transit light curve. Each one of these data files should have three columns themselves: time (in BJD), flux (on some arbitrary scale - TransitFit will normalise these to 1), and uncertainty on each flux measurement. The epoch and filter entries are indices which allow light cuves to be grouped in time or wavelength. These indices are zero-indexed.

### Priors file
This file allows the user to specify prior values, as well as which parameters are to fitted, and to set fixed values for other parameters. The 


<a name="limb_darkening"></a>
## Limb darkening

<a name="detrending"></a>
## Detrending


NOTE: The below here is legacy and will be removed eventually. I've just kept it for copy-pasting for now.

<a name="priors"></a>
## Setting Priors

Before running retrieval, the priors must be set up. This requires the creation of a ``PriorInfo`` object. There are two ways to do this.

### Method 1: create ``PriorInfo`` and add fitting parameters
The first method involves creating a ``PriorInfo`` object and setting default value for all parameters, and then adding bounds to parameters you want to fit. To create a `PriorInfo` object, you can use
```python
  prior_info = transitfit.setup_priors(args)
```
The docstring of ``setup_priors`` explains the full usage of this and the arguments. Once you have the ``PriorInfo`` set up, you can add a parameter to be fitted over using
```python
  prior_info.add_uniform_fit_param(name, best, low_lim, high_lim, light_curve_num=None)
```
The complete list of parameters and the names recognised by TransitFit are:

- ``'P'``: Period of the orbit

- ``'rp'``: Planet radius (in stellar radii)

- ``'t0'``: time of inferior conjunction

- ``a'`` : semi-major axis (in units of stellar radii)

- ``'inc'``: inclination of the orbit

- ``'ecc'``: eccentricity of the orbit

- ``'w'``: longitude of periastron (in degrees)



### Method 2: create ``PriorInfo`` from a .csv file

If you don't want to mess around with directly coding everything, TransitFit can read in your priors from a .csv file! To do this, use
```python
prior_info = transitfit.read_priors_file('/path/to/file/')
```
The docstring will tell you more about how this should be presented.

<a name="retrieval"></a>
## Running Retrieval

To run retrieval, set up a ``Retriever`` object
```python
retriever = transitfit.Retriever()
```
You also need to have your light curves. These need to be laid out as 3 (M,N) arrays, one for each of the time series, flux from the star (normalised), and the uncertainty on the flux. In the shape, M is the number of light curves you want to run retrieval on simultaneously. Note that these must all be for the same planet!! You can also use
```python
times, flux, errors = transitfit.read_data_file('/path/to/file') 
```
to read in a single light curve!

Now we can use the ``PriorInfo`` and ``Retriever`` together to get some results
```python
results = retriever.run_dynesty(times, flux, errors, prior_info)
```

<a name="single_line"></a>
## Single line running
If you have a bunch of data files and a prior.csv file set up, ``TransitFit`` provides a helpful wrapper function ``run_retrieval_from_paths()`` which does all of the above for you. All you have to do is pass it a list of paths to your data and the path to the priors file:

```python
import transitfit

data_file_paths = ['/data/path/1', 
                   '/data/path/2',
                   '/data/path/3'...]
                   
results = transitfit.run_retrieval_from_paths(data_file_paths, '/path/to/prior/file')
```

