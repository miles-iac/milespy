.. MILESpy documentation master file, created by
   sphinx-quickstart on Tue May 21 12:03:08 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MILESpy documentation
===================================

Welcome to the documentation for MILESpy - a python interface to the
`MILES <http://miles.iac.es>`_ SSP models and stellar library.
This package aims to provide users an easy interface to access single stellar
population (SSP) models, navigate the stellar library or synthesize the
spectrum of a given star formation history (SFH), among other things.
MILESpy is fully integrated and builds upon previously existing tools, namely
`astropy <https://www.astropy.org/>`_ and `specutils <https://specutils.readthedocs.io>`_.

We recommend you get started with the MILESpy :doc:`tutorials/index` to understand
the possibilities of the package. Then, navigate to the :doc:`installation` instructions
to start using the code. Lastly, during your development process, the :doc:`reference/index`
may be the best resource.

.. attention::
   Note that MILESpy is shipped with a simple interpolating routine for stellar
   spectra based on a Voronoi tesselation of the parameter space.
   We simply weight by the distance to the vertices of the enclosing cell. This
   is NOT the classical interpolator :cite:p:`Vazdekis2003a` used in
   :cite:t:`Vazdekis2010b` to generate SSP spectra from the MILES stellar
   library.

Available stellar libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::


   * - **Stellar library**
     - **Reference**
   * - **MILES**
     - :cite:t:`Sanchez-Blazquez2006`
   * - **EMILES**
     - :cite:t:`Koleva2012`
   * - **CaT**
     - :cite:t:`Cenarro2001`
   * - **sMILES**
     - :cite:t:`Knowles2021`

Available SSP models
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 2 3 3 3 2 3

   * - **SSP models**
     - **Wavelength coverage**
     - **Abundance pattern**
     - **IMF**
     - **Isochrones**
     - **Reference**
   * - **MILES** (base models)
     - 3400 - 7300 Å
     - Solar neighborhood
     - * Chabrier
       * Kroupa universal
       * Kroupa revised
       * unimodal
       * bimodal
     - * Padova00
       * BaSTI
     - :cite:t:`Vazdekis2010b`
   * - **MILES** (α-variable models)
     - 3400 - 7300 Å
     - [α/Fe]= 0.0; +0.4
     - * Chabrier
       * Kroupa universal
       * Kroupa revised
       * unimodal
       * bimodal
     - * BaSTI
     - :cite:t:`Vazdekis2015a`
   * - **EMILES**
     - 1680 - 50000 Å
     - Solar neighborhood
     - * Chabrier
       * Kroupa universal
       * Kroupa revised
       * unimodal
       * bimodal
     - * Padova00
       * BaSTI
     - :cite:t:`Vazdekis2016b`
   * - **CaT**
     - 8348 - 8950 Å
     - Solar neighborhood
     - * Kroupa universal
       * Kroupa revised
       * unimodal
       * bimodal
     - * Padova00
     - :cite:t:`Vazdekis2003a`


Please refer to the `official MILES website <https://research.iac.es/proyecto/miles/pages/ssp-models.php>`_ for more details. Also check the relevant publications for each of the models in :doc:`references`.

.. toctree::
   :maxdepth: 2

   Installation <installation>
   Configuration <configuration>
   Units handling <units>
   Tutorials & examples <tutorials/index>
   API Reference <reference/index>
   Contributing <development>
   Relevant publications <references>

.. bibliography::
