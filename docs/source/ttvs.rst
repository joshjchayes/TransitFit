=================
Allowing for TTVs
=================

In its default mode, ``TransitFit`` assumes that there are no TTVs and fits one value of :math:`t_0` to all transits. In the situation where TTVs are present, ``TransitFit`` can fit :math:`t_0` to each individual epoch, allowing then for O-C investigations.

In order to do this, set ``allow_TTV=True`` in the arguments of :meth:`~transitfit.run_retrieval`. **Note**: ``TransitFit`` cannot automatically detect if there are TTVs present in the data. You must explicitly enable this mode.

When ``allow_TTV=True``, ``TransitFit`` cannot fit for the period value, and this must be set as a fixed value in the :ref:`data input file <Data input file>`. In order to be consistent, we recommend that any investigation into TTVs in a system should have 2 stages:

1. Run ``TransitFit`` with ``allow_TTV=False``.

2. Run ``TransitFit`` with ``allow_TTV=True``, fixing the period at the best fit value from the first step.
