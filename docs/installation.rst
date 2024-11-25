Installation
============

In a standard python installation using pip as the package manager, just do:

.. code-block:: bash

  python3 -m pip install milespy

If you are installing from source, after cloning this repository, install it with:

.. code-block:: bash

  python3 -m pip install .

.. note::
  For most of the functionality of MILESpy you will need the repository
  data files. These contains all the stellar and SSP spectra, together with its
  associated metadata.

  The first time that you try to initialize MILESpy, you will be prompted if you
  want the repositories to be downloaded. You can change if and where these files
  are saved changing the `MILESpy configuration <configuration>`_.
