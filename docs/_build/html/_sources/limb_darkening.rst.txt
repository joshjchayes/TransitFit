==============
Limb-darkening
==============

``TransitFit`` was built with two primary motivations. First, to facilitate transmission spectroscopy surveys using observations from heterogeneous telescopes, and second, to allow the user to fit light curves while accounting for the effects that filter profiles and host parameters have on the LDCs, which we refer to as 'coupling' the LDCs. We will discuss the latter here.

The full justification for and impacts of this approach can be found in `the paper LINK TO COME <DEAD>`_ but, in short, not including host parameters and filter profiles in likelihood calculations of LDCs can lead to biases in your measurements of :math:`R_p/R_\star` of tens of percent. By including this, ``TransitFit`` has made it easy to conduct robust transmission spectroscopy studies using observations from heterogeneous sources.

Calculating LDC likelihoods
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TransitFit`` uses the `Limb Darkening Toolkit (LDTk) <https://github.com/hpparvi/ldtk>`_ to calculate the likelihood values of sets of LDCs given the host characteristics and filter profiles. These values are then included in the likelihood calculations of transit models during retrieval.

In order to use this feature, the user must provide the following:

    * Host temperature
    * Host :math:`z`
    * Host :math:`\log(g)`
    * A filter profile for each filter used in the observations.

    The first three are provided as arguments to :meth:`~transitfit.run_retrieval`, and the filter profiles are specified using the :ref:`filter profiles input file <Filter profiles>`.


Available  limb-darkening models
--------------------------------

Typically, stellar intensity profiles are described by analytical functions :math:`I_\lambda\left(\mu\right)`, where :math:`\mu` is the cosine of the angle between the line of sight and the emergent intensity. :math:`\mu` can also be expressed as :math:`\mu = \sqrt{1-r^2}` where :math:`r` is the unit-normalised radial coordinate on the stellar disk, and as such, all limb-darkening models must be valid for :math:`0 \le \mu < 1`.

There are 5 limb darkening models provided by ``TransitFit``, which can be selected using the ``limb_darkening_model`` argument in :meth:`~transitfit.run_retrieval`. These are:

    * ``'linear'`` - the linear law given by
        .. math::
            \frac{I\left(\mu\right)}{I\left(1\right)} = 1 - u_{0,l} \left(1 - \mu\right)

    * ``'quadratic'`` - the quadratic law given by
        .. math::
            \frac{I\left(\mu\right)}{I\left(1\right)} = 1 - u_{0,q} \left(1 - \mu\right) - u_{0,q} \left(1-\mu\right)^2

    * ``'squareroot'`` - the square-root law given by
        .. math::
            \frac{I\left(\mu\right)}{I\left(1\right)} = 1 - u_{0,\textrm{sqrt}} \left(1 - \mu\right) - u_{1,\textrm{sqrt}} \left(1-\sqrt{\mu}\right)

    * ``'power2'`` - the power-2 law given by
        .. math::
            \frac{I\left(\mu\right)}{I\left(1\right)} = 1 - u_{0,\textrm{p2}}\left(1 - \mu^{u_{1,\textrm{p2}}}\right)


    * ``'nonlinear'`` - the non-linear law given by
        .. math::
            \begin{split}
                \frac{I\left(\mu\right)}{I\left(1\right)} = 1 & - u_{0,\textrm{nl}} \left(1 - \mu^{1/2}\right) - u_{1,\textrm{nl}} \left(1-\mu\right) \\
                &- u_{2,\textrm{nl}} \left(1-\mu^{3/2}\right) - u_{3,\textrm{nl}} \left(1-\mu^{2}\right).
            \end{split}

where each of :math:`u_0`, :math:`u_1`, :math:`u_2`, and :math:`u_3` are the limb-darkening coefficients to be fitted. With the exception of the non-linear law, all of these models are constrained to physically-allowed values by the method in `Kipping (2013) <https://arxiv.org/abs/1308.0009>`_, which we have extended to include the power-2 law.

LDC Fitting modes
-----------------

``TransitFit`` offers three modes for LDC fitting, which can be selected using the ``ld_fit_method`` argument in :meth:`~transitfit.run_retrieval`.:

* ``'independent'``
    This is the traditional approach of fitting LDCs for each filter separately. ``TransitFit`` still uses the `Kipping parameterisations <https://arxiv.org/abs/1308.0009>`_, but LDTk is not used to couple LDCs across filters.

* ``'coupled'``
    Using the Kipping parameterisations, each LDC is fitted as a free parameter, with LDTk being used to estimate the likelihood of sets of LDCs, using information on the host star and the observation filters.

``'single'``
    When fitting with multiple wavebands, the number of parameters required to be fitted can increase dramatically. The ``'single'`` LDC fitting mode freely fitting LDC for only one filter, and uses LDTk to extrapolate LDC values for the remaining filters. The :math:`i`-th coefficient of a filter :math:`f`, is calculated as

    .. math::
        c_{i, f} = u_i \times \frac{\langle c_{i, f}\rangle}{\langle u_{i}\rangle}

    where :math:`u_i` is the sampled value of the :math:`i`-th LDC in the actively fitted filter, and :math:`\langle c_{i, f}\rangle` and :math:`\langle u_{i}\rangle` are the maximum likelihood values initially suggested by LDTk.
