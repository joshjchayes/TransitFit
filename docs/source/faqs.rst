====
FAQs
====

Why do I have to use config files? Isn't that a bit outdated?
    Fair point. We've based this around config files because when being used on hundreds of light curves at once, typing all the inputs into a list becomes very difficult to read and keep track of. In future updates, we might work on streamlining the API to allow inputs to be set directly within the code by the user.

How do I cite ``TransitFit``?
    If you have found ``TransitFit`` useful in your work, please cite `OUR PAPER-LINK TO ASDABS <DEAD>`_. If you're using BibTeX, you can copy the following into your .bib file::

        BibTeX CODE HERE

Why have you only included nested sampling? Why can't I use MCMC?
    This comes down partly to the personal preference of the development team, but mostly because ``TransitFit`` often has to deal with high-dimensioned fitting problems. MCMC routines often struggle in this situation, especially when the posterior space can be fairly degenerate or spiky. Nested sampling can handle these situations in a more stable way.

I've found a bug - what can I do?
    Raise the issue on the `GitHub page <https://github.com/joshjchayes/TransitFit>`_. Make sure to include as much information as possible, including any traceback messages and information on your priors etc.

Can I contribute to the project?
    Absolutely! ``TransitFit`` is an open-source project (GPL-3.0 licence) - please raise a pull request for any additions, changes, or improvements want to suggest.
