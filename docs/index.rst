.. pymiles documentation master file, created by
   sphinx-quickstart on Tue May 21 12:03:08 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pymiles documentation
===================================

Welcome to the documentation for pymiles - an python interface to the MILES
stellar library and SSP models. [#f]_
This package aims to provide users an easy interface to generate single stellar
population (SSP) models, navigate the stellar library or generate a spectra given
an input star formation history (SFH), among other things.
We try to make this package compatible with previously existing tools, namely
`astropy <https://www.astropy.org/>`_ and `specutils <https://specutils.readthedocs.io>`_.

We recommend you get started with the pymiles :doc:`tutorials/index` to understand
the possibilities of the package. Then, navigate to the :doc:`installation` instructions
to start using the code. Lastly, during your development process the :doc:`reference/index`
may be the best resource.

.. [#f] See [V10]_ and others (TO BE FILLED)
.. [V10] `Vazdekis et al. 2010; Evolutionary stellar population synthesis with MILES – I. The base models and a new line index system <https://academic.oup.com/mnras/article/404/4/1639/1080511>`_

.. toctree::
   :maxdepth: 2

   Installation <installation>
   Tutorials & examples <tutorials/index>
   Reference <reference/index>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
