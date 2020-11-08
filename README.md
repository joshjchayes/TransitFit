
# TransitFit

**Author** - Joshua Hayes (joshjchayes@gmail.com)

## Contents

[Overview](#overview)

[Requirements](#requirements)

[Basic Usage](#basics)

[Inputs](#inputs)

[Limb darkening](#limb_darkening)

[Detrending](#detrending)

[Fitting Large Numbers of Light Curves](#lots_of_curves)

[Citing TransitFit](#citing)


<a name="overview"></a>
## Overview
TransitFit is designed for exoplanetary transmission spectroscopy studies and offers a flexible approach to fitting single or multiple transits of an exoplanet at different observation wavelengths.  It possesses the functionality to efficiently couple host limb-darkening parameters to a range of physical models across different wavelengths, through the use of the [Limb darkening toolkit (ldtk)](https://github.com/hpparvi/ldtk) and the [Kipping parameterisations of two-parameter limb darkening models](https://arxiv.org/abs/1308.0009). TransitFit uses [batman](https://www.cfa.harvard.edu/~lkreidberg/batman/index.html) to handle transit light curve modelling, and sampling and retrieval uses the nested sampling algorithm available through [dynesty](https://dynesty.readthedocs.io/en/latest/index.html).

TransitFit can fit for physical parameters, including orbital period, planet radius, and limb darkening parameters. Alongside, it is also possible to fit for systematics and perform detrending simulateously with fitting of physical parameters. The detrending can be either through a polynomial of user-specified order, or a user-supplied function. More information can be found [here](#detrending).

In order to fit large numbers of light curves for a single planet, TransitFit offers a variety of fitting modes, which batch and fold light curves to reduce the number of parameters being simultaneously fitted for. This functionality can be turned off if desired, but this should be done with caution, as retrieval in very high dimensionality can be unstable.

User input is handled through external files (see [Input files](#inputs)), and is designed to be as intuitive as possible whilst still offering the flexibility which TransitFit was intended to have.


<a name="requirements"></a>
## Requirements
Along with an installation of Python 3 (with the standard Conda distribution packages), TransitFit requires the following packages to be installed:

- [dynesty](https://dynesty.readthedocs.io/en/latest/index.html)

- [batman](https://www.cfa.harvard.edu/~lkreidberg/batman/index.html)

- [Limb darkening toolkit (ldtk)](https://github.com/hpparvi/ldtk)


<a name="basics"></a>
## Basic Usage
Using TransitFit is as simple as calling a single function! You will need to have set up some input files (see [below](#inputs)). To fit a single lightcurve with basic linear detrending, it is as simple as running

```
import transitfit as tf

results = tf.run_retrieval(path/to/data_file, path/to/priors_file)
```

TransitFit is capable of more, including simultaneous multi-wavelength, multi-epoch fitting. To use this, we still use `tf.run_retrieval()`, but have to specify a few more arguments, including information on the host and the observation filters. In this example here, we will use some arbitrary values for the host parameters. Note that they are all provided as tuples of ``(value, error)``.

```
import transitfit as tf

# Paths to data, priors, and filter info:
data = '/path/to/data_file'
priors = '/path/to/priors_file'
filters = '/path/to/filter_info'

# host info
host_T = (5450, 130) # T in kelvin
host_z = (0.32, 0.09) 
host_r = (1.03, 0.05) # R in solar radii - this MUST be supplied if the prior for orbital separation is in AU. 
host_logg = (4.5, 0.1) # g in cm/s2

# Let's assume we want quadratic detrending - can set this with the detrending_list
detrending = [['nth order', 2]]

# Run the retrieval
results = tf.run_retrieval(data, priors, filters, detrending, host_T=host_T, host_z=host_z, host_logg=host_logg)
```

TransitFit also offers a variety of fitting modes, which can be used when fitting large numbers of light curves. These include folding the light curves, or fitting in filter batches. For more information, see [here](#lots_of_curves)


<a name="inputs"></a>
## Inputs
First, on light curve data. Each curve should be a separate text (.txt or .csv) file, with three columns: time (in BJD), flux, and uncertainty on flux. Note that the light curves do not have to be normalised or detrended, as this is something that TransitFit can handle.

User-supplied TransitFit inputs are primarily passed through .csv files, which are handled under the hood by TransitFit. The three files required are the data path file, which contains paths to the light curve data to be fitted, the priors file, which contains the priors for fitting, and, if coupling limb darkening across wavelengths, the filter info file, which defines the wavelengths of the observations.

#### Indexing
In order to allow for all possible combinations of light curves, and to offer flexibility on how each is detrended, TransitFit runs on an indexing system, which is zero-indexed. Each light curve requires a telescope, filter, and epoch index, and light curves can share one or two indices. In this way, it is possible to fit (for example) two simultaneous observations with the same filter from two different sites, or single wavelength, multi-epoch observations from a single observatory, or any other combination you can think of. We should note that the actual ordering of these indices does not matter, as long as they are consistent across all light curves being fitted, and no values are skipped.

Each light curve also requires a detrending index. This allows different detrending models to be used for different light curves, and can be useful in a situation where space- and ground-based observations are being combined (since space-based observations often require something more complex than an nth-order approach). More on detrending can be found [here](#detrending).

What follows is a quick overview of how to format each of the required files. TransitFit assumes that each of these contains one header row.

### Data paths
The data paths file should have 5 columns, in the order \[Path, telescope index, filter index, epoch index, detrending index\]

- **Path**: the absolute or relative path to the data file for each light curve.

- **Telescope index**

- **Filter index**

- **Epoch index**

- **Detrending index**


### Priors file
The priors file is used for defining which physical parameters are to be fitted, and the distribution from which samples are drawn. This file should have 5 columns, in the order \[Parameter, distribution, input A, input B, filter index\].

- **Parameter**: The parameters which can be set using the priors file are:

    - ``P``: Period of the orbit, in BJD

    - ``rp``: Planet radius in stellar radii (i.e. Rp/R*)

    - ``t0``: time of inferior conjunction in BJD

    - ``a`` : semi-major axis. This can be given in units of either host-radii or AU. If given in AU, then `host_r` must be specified in `tf.run_retrieval()` to allow for a conversion to host-radii.

    - ``inc``: inclination of the orbit in degrees

    - ``ecc``: eccentricity of the orbit

    - ``w``: longitude of periastron (in degrees)

    - {``q0``, ``q1``, ``q2``, ``q3``} : Kipping q parameters for limb darkening. Most of the time you will not need to set these, but if you want to run a retrieval without fitting for limb darkening (if, for example, you fitted for these another way), then you can set them here by specifying a ``'fixed'`` distribution. Note that you will also have to set ``ld_fit_method='off'`` in the arguments of ``run_retrieval()``.

- **distribution**: can be any of

    - ``uniform``: uses a uniform prior

    - ``gaussian``: uses a Gaussian prior

    - ``fixed``: This parameter won't be fitted and will be fixed at a value

- **Input A**: The use of this input varies depending on the distribution being used:

    - uniform: This is the lower bound of the uniform distribution

    - gaussian: this is the mean of the Gaussian distribution

    - fixed: this is the value to fix the parameter at

- **Input B**: the use of this input varies depending on the distribution being used:

    - uniform: This is the upper bound of the uniform distribution

    - gaussian: this is the standard deviation of the Gaussian distribution

    - fixed: not used

- **filter index**: Since planet radius varies with wavelength, the filter index must be supplied for each instance of planet radius in the priors file, making sure to follow the indexing set out in the data paths and filter info files.


### Filter info
This is used to specify the filter profiles that observations were made at. TransitFit can deal with either uniform box filters or user-specified profiles, and this file is used to determine this behaviour. This file should have three columns, in the order \[Filter index, low wavelength or path, high wavelength\]

- **Filter index**: the index of the filter, ensuring consistency with the other input files.

- **Low wavelength OR path to filter profile**: EITHER the lowest wavelength not blocked by the filter in nm (if uniform box filter) OR the path to the filter profile.

- **High wavelength**: the highest wavelength not blocked by the filter in nm. Ignored if filter profile provided.

If using a filter profile, this should have 2 columns - the wavelength and the fraction of light which can pass. Anything outside of the range of the profile is assumed to have a transmission of 0.

<a name="limb_darkening"></a>
## Limb darkening
The following limb darkening models are implemented in TransitFit, given here with the value to pass to `'limb_darkening_model'` in `tf.run_retrieval()`:

- The linear law - `'linear'`

- The quadratic law - `'quadratic'`

- The square root law - `'squareroot'`

- The power-2 law - `'power2'`. Note that this is only implemented when using the version of LDTk from the GitHub repo, rather than the `pip` installable version.

- The non-linear law - `'nonlinear'`

With the exception of the non-linear model, all models are constrained by the method in [Kipping (2013)],(https://arxiv.org/abs/1308.0009).

As discussed in more detail in the paper, TransitFit offers three modes of limb darkening coefficient fitting. These are (along with the value to give to `'ld_fit_method'`):

- **Independent** (`'independent'`) -  This is the traditional approach of fitting LDCs for each filter separately. TransitFit still uses the Kipping parameterisations, but LDTk is not used to couple LDCs across filters.

- **Coupled** (`'coupled'`) - Using the Kipping parameterisations, each LDC is fitted as a free parameter, with LDTk being used to estimate the likelihood of complete sets of LDCs, using information on the host star and the observation filters.

- **Single filter** (`'single'`) -  When fitting with multiple wavebands, the number of parameters required to be fitted can increase dramatically when using the coupled mode. Consequently, we have provided a method of only freely fitting the LDCs of one filter, and using LDTk to extrapolate LDC values for the remaining filters.

<a name="detrending"></a>
## Detrending
Detrending of light curves in TransitFit occurs in three different modes - `'nth order'`, `'custom'`, and `'off'`. Since observations being simultaneously fitted may require different treatments, the `detrending_list` kwarg in `run_retrieval()` allows this to be specified. `detrending_list` is an Nx2 iterable, where each entry is a pair of `[mode, info]`. The syntax for these is as follows

- `nth order` mode: `['nth order', X]`,  where X is the order of the detrending polynomial.

- `custom` mode: `['custom', f]`, where `f` is the custom detrending function. This can take any form but the first argument must be an array of times. 

- `off`: `['off', ]` - anything in the second entry will be ignored. 

To use different methods, use the detrending index specified in the data paths file to refer to the entry in `detrending_list`. 

<a name="lots_of_curves"></a>
## Fitting large numbers of light curves




<a name="citing"></a>
## Citing TransitFit
If you have used TransitFit in your work, please cite the accompanying paper (LINK TO FOLLOW):
```
SOME BIBTEX HERE
```
