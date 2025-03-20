Contributing to MILESpy
=======================

The following text has been loosely based on `astropy`'s [contribution
guidelines](https://github.com/astropy/astropy/blob/main/CONTRIBUTING.md)

Reporting Issues
----------------

When opening an issue to report a problem, please try to provide a minimal code
example that reproduces the issue along with details of the operating
system and the Python, NumPy, `astropy`, `spectutils` and `milespy` versions you are using.

Contributing Code and Documentation
-----------------------------------

If you are interesting in contributing to MILESpy you are lucky, we are open
to contributions! MILESpy is open source, and can be improved by people like you.

How to Contribute, Best Practices
---------------------------------

To contribute to MILESpy, the most efficient way is via a [pull
request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)
from GitHub users' forks of the [MILESpy
repository](https://github.com/miles-iac/milespy).

You may also/instead be interested in contributing to the environment in which
MILESpy is developed: `astropy`. In that case, please refer to their
[contribution
guide](https://github.com/astropy/astropy/blob/main/CONTRIBUTING.md)

Once you open a pull request (which should be opened against the ``main``
branch, not against any of the other branches), please make sure to
include the following:

- **Code**: the code you are adding, which should follow standard formatting as
  enforced in the pre-commit hooks.

- **Tests**: these are usually tests to ensure code that previously
  failed now works (regression tests), or tests that cover as much as possible
  of the new functionality to make sure it does not break in the future.

- **Documentation**: if you are adding new functionality, please add the proper
  documentation both in-code, and in the `docs` folder if it is far reaching and
  complex. Also, additional python notebooks examples can be added in `docs/tutorials`.


Checklist for Contributed Code
------------------------------

Before being merged, a pull request for a new feature will be reviewed to see if
it meets the following requirements.

**Code Quality**
  * Is the code compatible with the supported versions of Python (see [pyproject.toml](https://github.com/astropy/astropy/blob/main/pyproject.toml))?
  * Are there dependencies other than the run-time dependencies listed in pyproject.toml?

**Testing**
  * Are the inputs to the functions sufficiently tested?
  * Are there tests for any exceptions raised?
  * Does ``make tests`` run without failures?

**Documentation**
  * Is there a docstring in the function describing:
    * What the code does?
    * The format of the inputs of the function?
    * The format of the outputs of the function?
    * References to the original algorithms?
    * Any exceptions which are raised?
  * Is there any information needed to be added to the docs to describe the
    function?
  * Does the documentation build (i.e., `make doc`) without errors or warnings?


Other Tips
----------

- Behind the scenes, we conduct a number of tests or checks with new pull requests.
  This is a technique that is called continuous integration, and we use GitHub Actions.
  To prevent the automated tests from running, you can add ``[no test]``
  to your commit message. This is useful if your PR is a work in progress (WIP) and
  you are not yet ready for the tests to run. For example:

      $ git commit -m "[no test] WIP widget"

  - If you already made the commit without including this string, you can edit
    your existing commit message by running:

        $ git commit --amend

- If your commit makes substantial changes to the documentation but none of
  those changes include code, then you can use ``[no test]``, which will skip
  all CI code testing, except where the documentation is built.

- When contributing trivial documentation fixes (i.e., fixes to typos, spelling,
  grammar) that don't contain any special markup and are not associated with
  code changes, please include the string ``[no test]`` in your commit
  message.

      $ git commit -m "[no test] Fixed typo"

- Similarly, if your changes modify only the code without affecting the
  documentation, you can skip the documentation generation by including in the
  commit message ``[no docs]``.
