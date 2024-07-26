# -*- coding: utf-8 -*-
import numpy as np


def interp_weights(xyz, uvw, tri):
    """
    Creates a Delaunay triangulation and finds the vertices and weights of
    points around a given location in parameter space
    """

    d = len(uvw[0, :])
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum("njk,nk->nj", temp[:, :d, :], delta)

    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


# ==============================================================================
#
# NAME:
#   LOG_REBIN
#
# MODIFICATION HISTORY:
#   V1.0.0: Using interpolation. Michele Cappellari, Leiden, 22 October 2001
#   V2.0.0: Analytic flux conservation. MC, Potsdam, 15 June 2003
#   V2.1.0: Allow a velocity scale to be specified by the user.
#       MC, Leiden, 2 August 2003
#   V2.2.0: Output the optional logarithmically spaced wavelength at the
#       geometric mean of the wavelength at the border of each pixel.
#       Thanks to Jesus Falcon-Barroso. MC, Leiden, 5 November 2003
#   V2.2.1: Verify that lamRange[0] < lamRange[1].
#       MC, Vicenza, 29 December 2004
#   V2.2.2: Modified the documentation after feedback from James Price.
#       MC, Oxford, 21 October 2010
#   V2.3.0: By default now preserve the shape of the spectrum, not the
#       total flux. This seems what most users expect from the procedure.
#       Set the keyword /FLUX to preserve flux like in previous version.
#       MC, Oxford, 30 November 2011
#   V3.0.0: Translated from IDL into Python. MC, Santiago, 23 November 2013
#   V3.1.0: Fully vectorized log_rebin. Typical speed up by two orders of magnitude.
#       MC, Oxford, 4 March 2014
#   V3.1.1: Updated documentation. MC, Oxford, 16 August 2016


def log_rebin(lamRange, spec, oversample=False, velscale=None, flux=False):
    """
    Logarithmically rebin a spectrum, while rigorously conserving the flux.
    Basically the photons in the spectrum are simply redistributed according
    to a new grid of pixels, with non-uniform size in the spectral direction.

    When the flux keyword is set, this program performs an exact integration
    of the original spectrum, assumed to be a step function within the
    linearly-spaced pixels, onto the new logarithmically-spaced pixels.
    The output was tested to agree with the analytic solution.

    :param lamRange: two elements vector containing the central wavelength
        of the first and last pixels in the spectrum, which is assumed
        to have constant wavelength scale! E.g. from the values in the
        standard FITS keywords: LAMRANGE = CRVAL1 + [0,CDELT1*(NAXIS1-1)].
        It must be LAMRANGE[0] < LAMRANGE[1].
    :param spec: input spectrum.
    :param oversample: Oversampling can be done, not to loose spectral resolution,
        especally for extended wavelength ranges and to avoid aliasing.
        Default: OVERSAMPLE=1 ==> Same number of output pixels as input.
    :param velscale: velocity scale in km/s per pixels. If this variable is
        not defined, then it will contain in output the velocity scale.
        If this variable is defined by the user it will be used
        to set the output number of pixels and wavelength scale.
    :param flux: (boolean) True to preserve total flux. In this case the
        log rebinning changes the pixels flux in proportion to their
        dLam so the following command will show large differences
        beween the spectral shape before and after LOG_REBIN:

           plt.plot(exp(logLam), specNew)  # Plot log-rebinned spectrum
           plt.plot(np.linspace(lamRange[0], lamRange[1], spec.size), spec)

        By defaul, when this is False, the above two lines produce
        two spectra that almost perfectly overlap each other.
    :return: [specNew, logLam, velscale]

    """
    lamRange = np.asarray(lamRange)
    if len(lamRange) != 2:
        raise ValueError("lamRange must contain two elements")
    if lamRange[0] >= lamRange[1]:
        raise ValueError("It must be lamRange[0] < lamRange[1]")
    s = spec.shape
    if len(s) != 1:
        raise ValueError("input spectrum must be a vector")
    n = s[0]
    if oversample:
        m = int(n * oversample)
    else:
        m = int(n)

    dLam = np.diff(lamRange) / (n - 1.0)  # Assume constant dLam
    lim = lamRange / dLam + [-0.5, 0.5]  # All in units of dLam
    borders = np.linspace(*lim, num=n + 1)  # Linearly
    logLim = np.log(lim)

    c = 299792.458  # Speed of light in km/s
    if velscale is None:  # Velocity scale is set by user
        velscale = np.diff(logLim) / m * c  # Only for output
    else:
        logScale = velscale / c
        m = int(np.diff(logLim) / logScale)  # Number of output pixels
        logLim[1] = logLim[0] + m * logScale

    newBorders = np.exp(np.linspace(*logLim, num=m + 1))  # Logarithmically
    k = (newBorders - lim[0]).clip(0, n - 1).astype(int)

    specNew = np.add.reduceat(spec, k)[:-1]  # Do analytic integral
    specNew *= np.diff(k) > 0  # fix for design flaw of reduceat()
    specNew += np.diff((newBorders - borders[k]) * spec[k])

    if not flux:
        specNew /= np.diff(newBorders)

    # Output log(wavelength): log of geometric mean
    logLam = np.log(np.sqrt(newBorders[1:] * newBorders[:-1]) * dLam)

    return specNew, logLam, velscale


def log_unbinning(lamRange, spec, oversample=1, flux=True):
    """
    This function transforms logarithmically binned spectra back to linear
    binning. It is a Python translation of Michele Cappellari's
    "log_rebin_invert" function. Thanks to Michele Cappellari for his permission
    to include this function in the pipeline.
    """
    # Length of arrays
    n = len(spec)
    m = n * oversample

    # Log space
    dLam = (lamRange[1] - lamRange[0]) / (n - 1)  # Step in log-space
    # Min and max wavelength in log-space
    lim = lamRange + np.array([-0.5, 0.5]) * dLam
    borders = np.linspace(lim[0], lim[1], n + 1)  # OLD logLam in log-space

    # Wavelength domain
    # Min and max wavelength in Angst.
    logLim = np.exp(lim)
    lamNew = np.linspace(logLim[0], logLim[1], m + 1)  # new logLam in Angstroem
    newBorders = np.log(lamNew)  # new logLam in log-space

    # Translate indices of arrays so that newBorders[j] corresponds to borders[k[j]]
    k = np.floor((newBorders - lim[0]) / dLam).astype(int)
    # Construct new spectrum
    specNew = np.zeros(m)
    for j in range(0, m - 1):
        a = (newBorders[j] - borders[k[j]]) / dLam
        b = (borders[k[j + 1]] - newBorders[j + 1]) / dLam

        specNew[j] = np.sum(spec[k[j] : k[j + 1]]) - a * spec[k[j]] - b * spec[k[j + 1]]

    # Rescale flux
    if flux:
        specNew = (
            specNew
            / (newBorders[1:] - newBorders[:-1])
            * np.mean(newBorders[1:] - newBorders[:-1])
            * oversample
        )

    # Shift back the wavelength arrays
    lamNew = lamNew[:-1] + 0.5 * (lamNew[1] - lamNew[0])

    return specNew, lamNew
