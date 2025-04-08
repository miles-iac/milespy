.. MILESpy documentation master file, created by
   sphinx-quickstart on Tue May 21 12:03:08 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MILESpy documentation
===================================

Welcome to the documentation for MILESpy - a python interface to the
`MILES <http://miles.iac.es>`_ stellar library and SSP models.
This package aims to provide users an easy interface to generate single stellar
population (SSP) models, navigate the stellar library or generate a spectra given
an input star formation history (SFH), among other things.
We try to make this package compatible with previously existing tools, namely
`astropy <https://www.astropy.org/>`_ and `specutils <https://specutils.readthedocs.io>`_.

We recommend you get started with the MILESpy :doc:`tutorials/index` to understand
the possibilities of the package. Then, navigate to the :doc:`installation` instructions
to start using the code. Lastly, during your development process the :doc:`reference/index`
may be the best resource.

.. attention::
   Note that MILESpy is shipped with a simple interpolating routine based on a
   Voronoi tesselation of the parameter space. We simply weight by the distance
   to the vertices of the enclosing cell. This is NOT the classical
   interpolator :cite:p:`Vazdekis2003a` used in :cite:t:`Vazdekis2010b`
   to generate SSP spectra from the MILES stellar library.

Available stellar libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------+---------+-------+-----------------------------+
| Stellar library | Version | Notes | Reference                   |
+=================+=========+=======+=============================+
| MILES           | 9.1     |       | :cite:t:`Vazdekis2010b`     |
+-----------------+         +-------+-----------------------------+
| EMILES          |         |       |                             |
+-----------------+         +-------+-----------------------------+
| CaT             |         |       |                             |
+-----------------+---------+-------+-----------------------------+
| sMILES          |          in prep.                             |
+-----------------+---------+-------+-----------------------------+

Available SSP models
~~~~~~~~~~~~~~~~~~~~

+-----------+---------+----------------+-----------+-------+------------------------------+
| SSP model | Version | IMF            + Isochrone + Notes | Reference                    |
+===========+=========+================+===========+=======+==============================+
| MILES     | 9.1     |                |           |       | :cite:t:`Vazdekis2010b`      |
+-----------+         |                |           +-------+------------------------------+
| EMILES    |         | ch/ku/kb/un/bi | P/T       |       |                              |
+-----------+         |                |           +-------+------------------------------+
| CaT       |         |                |           |       |                              |
+-----------+---------+----------------+-----------+-------+------------------------------+
| sMILES    |          in prep.                                                           |
+-----------+---------+----------------+-----------+-------+------------------------------+

Where the IMF correspond to
  - ch: Chabrier
  - kb: Kroupa Revised
  - ku: Kroupa Universal
  - un: Unimodal with variable logarithmic slope
  - bi: Bimodal with variable massive stars segment logarithmic slope

And the isochrones to:
  - P: Padova+00
  - B: BaSTI

Please refer to the `official MILES website <https://research.iac.es/proyecto/miles/pages/ssp-models.php>`_ for more details. Also check the relevant publications for each of the models in :doc:`references`.

.. toctree::
   :maxdepth: 2

   Installation <installation>
   Configuration <configuration>
   Units handling <units>
   Tutorials & examples <tutorials/index>
   API Reference <reference/index>
   Relevant publications <references>

.. bibliography::
