Contributing to MILESpy
=======================

If you wish to add new functionality or test the latest version of MILESpy,
you can clone the repository directly:


.. code-block:: bash

  git clone https://github.com/miles-iac/milespy.git
  cd milespy


If you just wish to install this latest version (recommended to do in a new environment):

.. code-block:: bash

  python3 -m venv env
  . env/bin/activate
  python3 -m pip install .


But if you want to develop and contribute changes to MILESpy, the
best way is to use the provided
`Makefile <https://github.com/miles-iac/milespy/blob/main/Makefile>`_
to setup the environment.
Under the hood, it uses `uv <https://docs.astral.sh/uv/>`_
for environment management and dependency solving, so you need to
`install it <https://docs.astral.sh/uv/getting-started/installation/>`_.

Then, you can install MILESpy and all the required development dependencies:

.. code-block:: bash

  make install-dev
  . .venv/bin/activate

After you finish adding new functionalities, do not forget to run the tests
to check that your contribution does not break working bits of MILESpy. Also,
we highly encourage to add tests for newly added functionalities.

.. code-block:: bash

  make tests


If your contribution is only a fix for the documentation, you can ignore this steps,
and prepend to your commit message ``[no tests]`` to avoid running the test on GitHub
actions. Similarly, if the change does not modify the documentation, you can add
``[no docs]`` to the commit message.
