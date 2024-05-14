import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from pymiles.sfh import sfh as sfh

from scipy.integrate import simps

# ==============================================================================
if __name__ == '__main__':

    # -------------------------------------------------------------------------------
    # SSP MODELS EXAMPLES
    # -------------------------------------------------------------------------------
    plt.ioff()

    # Initializing instance
    print("# Initializing instance")
    sfh = sfh(source='MILES_SSP', version='9.1', imf_type='bi', isochrone='T', alp_type='variable', show_tree=False)

    # Let's play with the methods
    # First we select the age range we want
    sfh.set_time_range(start=13.5, end=0.01)

    # Then we define the SFR
    sfh.tau_sfr(start=11, tau=1.5, met=0.1)

    # Chemical and IMF evolution can also be included
    sfh.met_evol_sigmoid(start=-2.1, end=0.2, tc=10, gamma=2.)
    sfh.alp_evol_sigmoid(start=0.4, end=0.0, tc=10)
    sfh.imf_evol_linear(start=0.5, end=2.3, t_start=11.5, t_end=9.)

    # Let's see how it looks
    plt.plot(sfh.time, sfh.imf_evol, label='IMF slope')
    plt.plot(sfh.time, sfh.met_evol, label='[M/H]')
    plt.plot(sfh.time, sfh.sfr * 10, label='SFR (scaled)')
    plt.xlabel('Look-back time')
    plt.legend()
    plt.savefig('sfh.png')
    plt.close('all')

    # And finally some predictions
    pred = sfh.get_sfh_predictions()
    fnames = pred.find_filter("sloan")
    filts = pred.get_filters(fnames)
    outmls = pred.compute_ml(filters=filts, type='star+remn', verbose=False)

    print('Mass-weighted age', pred.age)
    print('Mass-weighted [M/H]', pred.met)
    print('M/L g-band', outmls['SLOAN_SDSS.g'])

    # And finally the spectra
    plt.plot(pred.wave, pred.spec)
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.legend()
    plt.tight_layout()
    plt.savefig('spec.png')
    plt.close('all')
