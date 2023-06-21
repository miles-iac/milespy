import numpy                 as np
import matplotlib.pyplot     as plt
from   pymiles.scripts.stellar_library_class import stellar_library as stl
#==============================================================================
if (__name__ == '__main__'):

# -------------------------------------------------------------------------------
# STELLAR LIBRARIES EXAMPLES
# -------------------------------------------------------------------------------

    # Initializing instance
    print("# Initializing instance")
    lib = stl(source='MILES_STARS',version='9.1')
#    lib = stl(source='CaT_STARS',version='9.1')

    # Get starname by ID
    print("# Getting starname by ID")
    starname = lib.get_starname(id=100)
    print(starname)

    # Searching by ID
    print("# Searching star data by ID")
    tmp = lib.search_by_id(id=100)
    print(tmp.starname,tmp.wave_init, tmp.wave_last)
    plt.plot(tmp.wave, tmp.spec)
    plt.title(tmp.starname[0])
    plt.show()

    # Get stars within parameter range
    print("# Getting stars within range")
    tmp = lib.get_stars_in_range(teff_lims=[4500.0,5000.0], logg_lims=[2.0,2.5], FeH_lims=[0.0,0.2])
    print("Teff",np.nanmin(tmp.teff),np.nanmax(tmp.teff))
    print("Log(g)",np.nanmin(tmp.logg),np.nanmax(tmp.logg))
    print("[Fe/H]",np.nanmin(tmp.FeH), np.nanmax(tmp.FeH))

    # Search by params (Gets the closest spectra to those params)
    print("# Search closest star to params")
    tmp = lib.search_by_params(teff=5000.0, logg=3.0, FeH=0.0, MgFe=0.0)
    print(tmp.id, tmp.teff, tmp.logg, tmp.FeH, tmp.MgFe)
    plt.plot(tmp.wave, tmp.spec)
    plt.title(tmp.starname)
    plt.show()
    
    """
    # Get spectrum by params (gets interpolated spectrum for those params)
    print("# Get spectrum by params (gets interpolated spectrum for those params)")
    tmp = lib.search_by_params(teff=5000.0, logg=3.0, FeH=0.0, MgFe=0.0)
    wave, spec = lib.get_spectrum_by_params_delaunay(teff=5000.0, logg=3.0, FeH=0.0, MgFe=0.0)
    plt.plot(tmp.wave, tmp.spec, label="Closest star")
    plt.plot(wave,spec,label="Interpolated star")
    plt.legend()
    plt.show()
    """
      