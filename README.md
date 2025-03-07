# milespy

![Action badge](https://github.com/miles-iac/milespy/actions/workflows/test.yml/badge.svg)

milespy is a python interface to the [MILES](http://miles.iac.es) stellar
library and SSP models.  This package aims to provide users an easy interface
to generate single stellar population (SSP) models, navigate the stellar
library or generate a spectra given an input star formation history (SFH),
among other things.  We try to make this package compatible with previously
existing tools, namely [astropy](https://www.astropy.org/) and
[specutils](https://specutils.readthedocs.io).

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
