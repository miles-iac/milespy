# milespy

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![PyPi](https://img.shields.io/pypi/v/milespy)](https://pypi.org/project/milespy)
![Test badge](https://github.com/miles-iac/milespy/actions/workflows/test.yml/badge.svg)
[![Docs badge](https://github.com/miles-iac/milespy/actions/workflows/docs.yml/badge.svg)](https://miles-iac.github.io/milespy/)
![Coverage Status](https://raw.githubusercontent.com/miles-iac/milespy/coverage-badge/coverage.svg?raw=true)

MILESpy is a python interface to the [MILES](http://miles.iac.es) stellar
library and SSP models.  This package aims to provide users an easy interface
to generate single stellar population (SSP) models, navigate the stellar
library or generate a spectra given an input star formation history (SFH),
among other things.  We try to make this package compatible with previously
existing tools, namely [astropy](https://www.astropy.org/) and
[specutils](https://specutils.readthedocs.io).

The documentation can be [browsed online](https://miles-iac.github.io/milespy/).
It has extensive [tutorials and examples](https://miles-iac.github.io/milespy/tutorials/index.html)
to easily start developing.

## Getting started

In a standard python installation using pip as the package manager, just do:

```bash
python3 -m pip install milespy
```

If you are installing from source, after cloning this repository, install it with:

```bash
python3 -m pip install .
```

## Support

If you find bugs or have feature suggestions, please submit an
[issue](https://github.com/miles-iac/milespy/issues) to this repository.
