Units handling
===================================

When working with scientific data is crucial to have in mind the different
units being used. This has been problematic when programming as each floating
point variable does not store the associated unit information.

To mitigate this problem in the astronomy community that uses python, `astropy
<https://www.astropy.org/>`_ provides a module to handle units in a transparent
way.  The key element of this module is the concept of a
:doc:`astropy:api/astropy.units.Quantity`.  It stores both numerical data (a
single number or a numpy array) and its unit.

In pymiles, we make extensive use of this mechanism to avoid unit conversion
mistakes and have a cleaner user interaface.  For example, most of the API
exposed to users check that the input units are correct (i.e., you can not pass
a mass to an input varible that expect an age).  In addition, you can provide
any input unit as long as it convertible to the correct one.  For example, you
can provide age in `Gyr`, `yr`, or seconds if you wish, and the output will be the
same.

We recommend you to take a look at the different `tutorials <tutorials/index.html>`_ available in this
documentation to see how to work with the units in pymiles.

.. note ::
   There are some special units in astropy used for logarithmic quantities. We
   make extensive use of those as the input metallicities should be in [M/H], for example.
   In that case, the unit to use is :doc:`dex <astropy:api/astropy.units.Dex>` (see :doc:`astropy:units/logarithmic_units`).
