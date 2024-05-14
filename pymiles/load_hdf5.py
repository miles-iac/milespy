# -*- coding: utf-8 -*-
import h5py
import numpy as np

# ==============================================================================


def load_hdf5(filename):
    # Initialise the output dictionary
    out = {}

    # Open the HDF5 file
    f = h5py.File(filename, "r")

    # Get the list of datasets
    list = f.keys()

    # Read the contents of the file and store them in the dictionary
    # NOTE: we skip the SPECTRA (the heavy part). We parse the string
    # values in a different way to get a proper np.array
    for key in list:
        if key == "spectra":
            continue

        if (key == "isochrone") or (key == "imf_type"):
            out[key] = np.array(np.array(f[key]), dtype="str")
        else:
            out[key] = np.array(f[key])

    return out


# ==============================================================================
if __name__ == "__main__":
    filename = "pyMILES/repository/EMILES_SSP_v9.1.hdf5"

    out = load_hdf5(filename)

    print(out.keys())

    exit()
