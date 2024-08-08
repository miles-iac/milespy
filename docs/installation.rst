Installation
============

In a standard python installation using pip as the package manager, just do:

.. code-block:: bash

  python3 -m pip install pymiles

If you are installing from source, after cloning this repository, install it with:

.. code-block:: bash

  python3 -m pip install .

.. note::
  For most of the functionality of pymiles you will need the repository
  data files. These contains all the stellar and SSP spectra, together with its
  associated metadata.

  The first time that you try to initialize pymiles, you will be prompted if you
  want the repositories to be downloaded. You can change if and where these files
  are saved changing the `pymiles configuration <configuration>`_.
