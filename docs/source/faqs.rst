====
FAQs
====

Why do I have to use config files? Isn't that a bit outdated?
    Fair point. We've based this around config files because when being used on hundreds of light curves at once, typing all the inputs into a list becomes very difficult to read and keep track of. In future updates, we might work on streamlining the API to allow inputs to be set directly within the code by the user.

How do I cite ``TransitFit``?
    If you have found ``TransitFit`` useful in your work, please cite `the accompanying paper <https://ui.adsabs.harvard.edu/abs/2021arXiv210312139H/abstract>`_. If you're using BibTeX, you can copy the following into your .bib file::

        @ARTICLE{2021arXiv210312139H,
               author = {{Hayes}, J.~J.~C. and {Kerins}, E. and {Morgan}, J.~S. and {Humpage}, A. and {Awiphan}, S. and {Charles-Mindoza}, S. and {McDonald}, I. and {Inyanya}, T. and {Padjaroen}, T. and {Munsaket}, P. and {Chuanraksasat}, P. and {Komonjinda}, S. and {Kittara}, P. and {Dhillon}, V.~S. and {Marsh}, T.~R. and {Reichart}, D.~E. and {Poshyachinda}, S.},
                title = "{TransitFit: an exoplanet transit fitting package for multi-telescope datasets and its application to WASP-127~b, WASP-91~b, and WASP-126~b}",
              journal = {arXiv e-prints},
             keywords = {Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
                 year = 2021,
                month = mar,
                  eid = {arXiv:2103.12139},
                pages = {arXiv:2103.12139},
        archivePrefix = {arXiv},
               eprint = {2103.12139},
         primaryClass = {astro-ph.EP},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210312139H},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }

Why have you only included nested sampling? Why can't I use MCMC?
    This comes down partly to the personal preference of the development team, but mostly because ``TransitFit`` often has to deal with high-dimensioned fitting problems. MCMC routines often struggle in this situation, especially when the posterior space can be fairly degenerate or spiky. Nested sampling can handle these situations in a more stable way.

I've found a bug - what can I do?
    Raise the issue on the `GitHub page <https://github.com/joshjchayes/TransitFit>`_. Make sure to include as much information as possible, including any traceback messages and information on your priors etc.

Can I contribute to the project?
    Absolutely! ``TransitFit`` is an open-source project (GPL-3.0 licence) - please raise a pull request for any additions, changes, or improvements want to suggest.
