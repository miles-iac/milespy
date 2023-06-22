import numpy             as np
import matplotlib.pyplot as plt
import h5py
from   astropy.io        import ascii, fits
from   pymiles.scripts.ssp_models_class  import ssp_models as ssp
from   pymiles.scripts.tuning_tools_class import tuning_tools as tt
#==============================================================================
if (__name__ == '__main__'):

# -------------------------------------------------------------------------------
# SSP MODELS EXAMPLES
# -------------------------------------------------------------------------------

    # Initializing instance
    print("# Initializing instance")
    miles = ssp(source='MILES_SSP', version='9.1', imf_type='bi', isochrone='P', alp_type='fix', show_tree=False)
    print('Nota: creo haber identificado que si pones la isocrona incorrecta, ciertas llamadas fallan.')
    

    # Get SSP in range ---------------------------------------------------
    miles_1 = miles.get_ssp_by_params(age=5.7, met=-0.45, imf_slope=1.3, return_pars=False)       
    plt.plot(miles_1.wave, miles_1.spec)
    plt.show()
    
    # Get SSP in range ---------------------------------------------------
    print("# Get SSP in range")
    miles_1 = miles.get_ssp_in_range(age_lims=[17.0,20.0], met_lims=[0.1,0.5], verbose=False)
    print(np.amin(miles_1.age),np.amax(miles_1.age))
    print(miles_1.age.shape)
    print(miles_1.age)
    print(miles_1.met)
    print(miles_1.Mass_star_remn)
#     exit()

    ###  Get SSP in list ---------------------------------------------------
    print("# Get SSP in list")
    miles_2 = miles.get_ssp_in_list(age_list=[0.2512,0.0708,1.4125], 
                                        met_list=[0.22,0.0,-1.71], 
                                        imf_slope_list=[1.3,1.3,1.3], 
                                        alpha_list=[0.0,0.4,0.0])

    
    # Get SSP by params (gets interpolated spectrum) ---------------------------------------------------
    # ("# Get SSP by params (gets interpolated spectrum)")
    miles_1 = miles.get_ssp_by_params(age=5.7, met=-0.45, imf_slope=1.3, return_pars=False)   
    print(miles_1.age)
    print(miles_1.filter_names) 
    
    fnames  = miles_1.find_filter("sloan")
    filts   = miles_1.get_filters(fnames)
    outmls  = miles_1.compute_ml(filters=filts, type='star+remn',verbose=False)
    
    print(miles_1.age, miles_1.met, miles_1.Mass_star_remn,
    outmls['SLOAN_SDSS.u'], outmls['SLOAN_SDSS.g'],outmls['SLOAN_SDSS.r'],
    outmls['SLOAN_SDSS.i'],outmls['SLOAN_SDSS.z'])   
    
    plt.plot(miles_1.wave, miles_1.spec)
    plt.show()
    # exit()
    
    #  Tune spectra ---------------------------------------------------
#     tab = ascii.read("../config_files/MUSE-WFM.lsf") # Hay que preparar para que la ruta funcione correctamente
    tab = ascii.read("./pymiles/config_files/MUSE-WFM.lsf")
    lsf_wave = tab['Lambda']
    lsf = tab['FWHM']*0.25
    
    tuned_miles = miles_1.tune_spectra(wave_lims=[4750.,5500.], dwave=1.0, sampling='lin',
                                        redshift=0.0, lsf_flag=True, lsf_wave=lsf_wave, lsf=lsf, verbose=False)
    
    plt.plot(miles_1.wave, miles_1.spec)   
    plt.plot(tuned_miles.wave, tuned_miles.spec, 'k')
    plt.show()
    
    print(tuned_miles.alpha)
    # Saving object to HDF5 file ---------------------------------------------------
    tuned_miles.save_object("./pymiles/saved_files/kk.hdf5", verbose=True)
    
    # Compute mags ---------------------------------------------------
    fnames  = miles_1.find_filter("sdss")
    filts   = miles_1.get_filters(fnames)
    outmags = miles_1.compute_save_mags(filters=filts, zeropoint='AB', saveCSV=False, verbose=True)  
    
    # Compute LS indices ---------------------------------------------------
    outls = miles_1.compute_ls_indices(verbose=True, saveCSV=False)   
    print(outls)
    
    # Compute solar mags (this can be done with the Tuning Tools class. No need for SSP class) ---------------------------------------------------
    fnames  = miles.find_filter("sloan")
    filts   = miles.get_filters(fnames)
    outmags = miles.compute_mag_sun(filters=filts, verbose=True, zeropoint='AB')  
    print(outmags)
    # exit()
    
    # Compute mass-to-light ratios ---------------------------------------------------
    print("# Computing M/Ls")
    fnames  = miles_1.find_filter("sloan")
    filts   = miles_1.get_filters(fnames)
    outmls  = miles_1.compute_ml(filters=filts, type='star+remn',verbose=False)
#     for i in np.arange(len(miles_1.age)):
#         print(miles_1.age[i], miles_1.met[i], miles_1.Mass_star_remn[i],
#         outmls['SLOAN_SDSS.u'][i], outmls['SLOAN_SDSS.g'][i],outmls['SLOAN_SDSS.r'][i], outmls['SLOAN_SDSS.i'][i],outmls['SLOAN_SDSS.z'][i]) !Revisar  

    """
    # IA: NOTE, THIS IS VERY SIMILAR TO examples_sfh ... is this here by mistake??
    # Extract SFHs ---------------------------------------------------
    print("# Creating SFH spectra and predictions")
    
    # First we select the age range we want
    miles.set_time_range(start=13.5, end=0.01)
    
    # Then we define the SFR
    print(" - Define input SFR")
    miles.tau_sfr(start=11, tau=1.5, met=0.1)
    
    # Chemical and IMF evolution can also be included
    print(" - Define chemical and IMF evolutio")
    miles.met_evol_sigmoid(start=-2.1,end=0.2,tc=10,gamma=2.)
    miles.alp_evol_sigmoid(start=0.4,end=0.0,tc=10)
    miles.imf_evol_linear(start=0.5,end=2.3,t_start=11.5, t_end=9.)
    
    # Let's see how it looks
    print(" - Plot results")
    plt.plot(miles.time,miles.imf_evol,label='IMF slope')
    plt.plot(miles.time,miles.met_evol,label='[M/H]')
    plt.plot(miles.time,miles.sfr*10,label='SFR (scaled)')
    plt.xlabel('Look-back time')
    plt.legend()
    plt.show()
    
    print("- Getting some predictions")
    # And finally some predictions
    pred   = miles.get_sfh_predictions() 
    fnames = pred.find_filter("sloan")
    filts  = pred.get_filters(fnames)
    outmls = pred.compute_ml(filters=filts, type='star+remn',verbose=False)
    
    print(' * Mass-weighted age',pred.age)
    print(' * Mass-weighted [M/H]', pred.met)
    print(' * M/L g-band', outmls['SLOAN_SDSS.g'])
    
    # And finally the spectra
    print(" - Plotting resulting spectra")
    plt.plot(pred.wave,pred.spec)
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.tight_layout()
    plt.show()
    """
