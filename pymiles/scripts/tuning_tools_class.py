import os
import re
import sys
import logging
import glob
import h5py
import warnings
import numpy               as np
import matplotlib.pyplot   as plt
import pymiles.scripts.pymiles_utils       as utils
import pymiles.scripts.cap_utils           as cap
import pymiles.scripts.misc_functions      as misc
from   astropy.io          import ascii, fits
from   copy                import copy
from   scipy               import interpolate
from astropy.units import mG
#==============================================================================

logger = logging.getLogger('pymiles.tuning_tools')

class tuning_tools:
   
   warnings.filterwarnings("ignore")  

   # solar_ref_spec = "./pymiles/config_files/sun_reference_stis_002.fits"
   solar_ref_spec = "./pymiles/config_files/sun_mod_001.fits"
   emiles_lsf     = "./pymiles/config_files/EMILES.lsf"
   lsfile         = "./pymiles/config_files/ls_indices_full.def"

# -----------------------------------------------------------------------------
# __INIT__
#
# Creates an instance of the class
# -----------------------------------------------------------------------------
   def __init__(self, wave_init=None, wave_last=None, dwave=None, 
                source=None, redshift=0.0, sampling='lin', 
                wave=None, spec=None):

      """
      Creates an instance of the class 
      
      Keywords
      --------
      wave_init:  Starting wavelength in Angstroms. If not defined, taken from WAVE
      wave_last:  End wavelength in Angstroms. If not defined, taken from WAVE
      dwave:      Wavelength step in Angstroms. If not defined, taken from WAVE
      source:     Name of input source to use. Valid inputs are MILES_STARS/CaT_STARS/MILES_SSP/CaT_SSP/EMILES_SSP. Default: MILES_SSP
      redshift:   Redshift of the input spectra. Default=0.0
      sampling:   Type of sampling of the spectra. Valid inputs are lin/ln. Default: lin
      wave:       Vector with input wavelengths in Angstroms
      spec:       [N,M] array with input spectra
    
      Return
      ------
      Object instance
    
      """
      logger.debug(wave)
      if (len(wave) == 0):
         wave = np.zeros(10)
      if (len(spec) == 0):
         spec = np.zeros((10,1))  

      if wave_init is None:
         self.wave_init = np.amin(wave)
      else:    
         self.wave_init = wave_init

      if wave_last is None:
         self.wave_last = np.amax(wave)
      else:       
         self.wave_last = wave_last

      if dwave is None:
         self.dwave = wave[1]-wave[0] 
      else:
         self.dwave = dwave

      self.redshift  = redshift
      self.sampling  = sampling
      self.wave      = np.array(wave)
      self.spec      = np.array(spec)
      self.npix      = spec.shape[0]
      self.nspec     = spec.shape[1]
      self.source    = source

      # Computing the LSF
#       if source != None:
#          self.compute_lsf()

      # Loading filters names
      fnames = glob.glob("./pymiles/config_files/filters/*.dat")
      self.filter_names = [os.path.basename(x).split(".dat")[0] for x in fnames]
      self.filter_names = np.sort(self.filter_names)
      self.nfilters = len(self.filter_names)

      # Checking inputs and redifining values if needed
      if np.amin(wave) < self.wave_init:
          wave_init = np.amin(wave)

      if np.amax(wave) > self.wave_last:
          wave_last = np.amax(wave)

      if len(wave) != self.npix:
          raise ValueError("Number of pixels in WAVE not equal to SPEC.")

      sampling_list = ['lin','ln']
      if sampling not in sampling_list:
          raise ValueError("SAMPLING has to be lin/ln")

      if redshift < 0.0:
          raise ValueError("REDSHIFT cannot be lower than 0.0")

      return

# -----------------------------------------------------------------------------
   def find_filter(self,name):

      """
      Searches for a filter in database.

      Note
      _____
      Search is case insensitive
      The filter seach does not have to be precise. Substrings within filter names are ok 
      It uses the python package 're' for regular expressions matching
      
      Arguments
      --------
      name: The search string to match filter names
    
      Return
      ------
      List of filter names available matching the search string
    
      """

      idx = np.zeros(self.nfilters, dtype='bool')
      for i in range(self.nfilters):
         check = re.search(name, self.filter_names[i], re.IGNORECASE)
         if check:
            idx[i] = True

      if np.sum(idx) == 0:
          logger.error("Cannot find filter in our database\n Available filters are:\n\n"+list(self.filter_names))
          sys.exit

      return list(self.filter_names[idx])

# -----------------------------------------------------------------------------
   def get_filters(self,filter_names):

      """
      Retrieves filter from database

      Arguments
      --------
      filter_names: The filter names
    
      Return
      ------
      Object with filter's wavelength and normalised transmission
    
      """

      nfilters = len(filter_names)
   
      filters = {}
      for i in range(nfilters):
         filename = "./pymiles/config_files/filters/"+filter_names[i]+".dat"
         if not os.path.exists(filename):
            logger.warning("Filter "+filter_names[i]+" does not exist in database")
            # sys.exit
         else:
            tab = ascii.read(filename, names=['wave','trans'])
            tab['trans'] /= np.amax(tab['trans'])
            filters[filter_names[i]] = tab

      return filters

# -----------------------------------------------------------------------------
   def plot_filters(self,filter_names, legend=True):

      """
      Plot filters

      Arguments
      --------
      filter_names: The filter names
      legend:       Flag to turn on/off the legend
    
      Return
      ------
      Nothing
    
      """

      nfilters = len(filter_names)
   
      for i in range(nfilters):
         filename = "./pymiles/config_files/filters/"+filter_names[i]+".dat"
         if not os.path.exists(filename):
            logger.warning("Filter "+filter_names[i]+" does not exist in database")
         else:
            tab = ascii.read(filename, names=['wave','trans'])
            tab['trans'] /= np.amax(tab['trans'])
            plt.fill_between(tab['wave'],tab['trans'],alpha=0.5, label=filter_names[i], edgecolor='k')

      if legend:
         plt.legend()        
      plt.show()

      return

# -----------------------------------------------------------------------------
   def update_basic_pars(self, wave, spec):

      """
      Updates basic values of the spectra in instance 
      
      Keywords
      --------
      wave:       Vector with input wavelengths in Angstroms
      spec:       [N,M] array with input spectra
    
      Return
      ------
      Object instance
    
      """

      self.wave_init = np.amin(wave)
      self.wave_last = np.amax(wave)
      self.dwave     = wave[1]-wave[0]
      self.wave      = wave
      self.spec      = spec
      self.npix      = spec.shape[0]
      self.nspec     = spec.shape[1]

      return

# -----------------------------------------------------------------------------
   def compute_lsf(self):

      """
      Returns the LSF given a source and wavelength from self
      
      Return
      ------
      Object instance with LSF info included
    
      """

      cvel = 299792.458
      self.lsf_wave = self.wave
      if self.source == 'MILES_SSP':
          self.lsf_fwhm  = 2.51*np.ones(self.npix)
          self.lsf_vdisp = cvel * (self.lsf_fwhm/2.355) / self.wave

      elif self.source == 'MILES_STARS':
          self.lsf_fwhm  = 2.50*np.ones(self.npix)
          self.lsf_vdisp = cvel * (self.lsf_fwhm/2.355) / self.wave

      elif self.source == 'CaT_SSP':
          self.lsf_fwhm  = 1.50*np.ones(self.npix)
          self.lsf_vdisp = cvel * (self.lsf_fwhm/2.355) / self.wave

      elif self.source == 'CaT_STARS':
          self.lsf_fwhm  = 1.50*np.ones(self.npix)
          self.lsf_vdisp = cvel * (self.lsf_fwhm/2.355) / self.wave

      elif self.source == 'EMILES_SSP':
          tab            = ascii.read(self.emiles_lsf) 
          wave           = tab['col1']
          fwhm           = tab['col2']
          sigma          = tab['col3']
          f_fwhm         = interpolate.interp1d(wave, fwhm)
          f_vdisp        = interpolate.interp1d(wave, sigma)
          self.lsf_fwhm  = f_fwhm(self.wave)
          self.lsf_vdisp = f_vdisp(self.wave)

      else:
          raise ValueError(self.source+" is not a valid entry. Allowed values: MILES_SSP/MILES_STARS/CaT_SSP/CaT_STARS/EMILES")

      return

# -----------------------------------------------------------------------------
   def trim_spectra(self, wave_lims=None, verbose=False):

      """
      Trims spectra to desired wavelength limits

      Keywords
      --------
      wave_lims: Wavelength limits in Angtroms
      verbose:   Flag for verbose
      
      Return
      ------
      Object instance with spectra trimmed and updated info
    
      """
      logger.info("# Trimming spectra in wavelength ...")

      out   = copy(self)
      idx   = (out.wave >= wave_lims[0]) & (out.wave <= wave_lims[1]) 
      wave  = out.wave[idx]
      spec  = out.spec[idx,:]
      out.update_basic_pars(wave,spec)
      out.compute_lsf()

      return out

# -----------------------------------------------------------------------------
   def resample_spectra(self, wave_lims=None, dwave=None, verbose=False):

      """
      Returns a copy of the instance with the wavelength vector 
      and spectra array rebinned to the desired wavelength step

      Keywords
      --------
      wave_lims: Desired wavelength limits in Angtroms
      dwave:     Desired wavelength step in Angstroms
      verbose:   Flag for verbose
      
      Return
      ------
      Object instance with spectra resampled and updated info
    
      """
      logger.info("# Resampling spectra ...")

      out      = copy(self)
      new_wave = np.arange(wave_lims[0],wave_lims[1],dwave)
      npix     = len(new_wave)
      spec     = np.zeros((npix,out.nspec))
      for i in range(out.nspec):
         spec[:,i] = utils.spectres(new_wave, out.wave, out.spec[:,i], fill=0.0)
         if verbose:
            misc.printProgress(i+1,out.nspec)
      out.update_basic_pars(new_wave,spec)
      out.compute_lsf() 

      return out

# -----------------------------------------------------------------------------
   def redshift_spectra(self, redshift=None, verbose=False):

      """
      Returns a copy of the instance with a redshifted wavelength vector, spectra and LSF

      Keywords
      --------
      redshift: Desired redshift
      verbose:  Flag for verbose
      
      Return
      ------
      Object instance with spectra redshifted and updated info
    
      """
      logger.info("# Redshifting spectra ...")

      out   = copy(self)
      wave  = out.wave * (1.0 + redshift)
      spec  = out.spec / (1.0 + redshift)
      out.update_basic_pars(wave, spec)
      out.redshift  = redshift
      out.lsf_wave  = wave
      out.lsf_fwhm  = out.lsf_fwhm  / (1.0 + redshift)
      out.lsf_vdisp = out.lsf_vdisp / (1.0 + redshift)

      return out

# -----------------------------------------------------------------------------
   def logrebin_spectra(self, velscale=None, verbose=False):

      """
      Returns a logrebinned version of the spectra

      Keywords
      --------
      velscale: Desired velocity scale in km/s. Computed automatically if None.
      verbose:  Flag for verbose
      
      Return
      ------
      Object instance with ln-rebinned spectra and updated info
    
      """
      logger.info("# Ln-rebining the spectra ...")

      if self.sampling == 'ln':
          logger.warning("Spectra already in ln-lambda.")
          return copy(self)
  
      out      = copy(self)
      lamRange = [out.wave_init,out.wave_last]
      lspec, lwave, velscale = cap.log_rebin(lamRange, out.spec[:,0], velscale=velscale)
      out_spec = np.zeros((len(lspec),out.nspec)) 
      for i in range(out.nspec): 
          out_spec[:,i], lwave, velscale = cap.log_rebin(lamRange, self.spec[:,i], velscale=velscale)
          if verbose:
             misc.printProgress(i+1,out.nspec)
      
      out.sampling = 'ln'
      out.update_basic_pars(lwave,out_spec)
  
      return out

# -----------------------------------------------------------------------------
   def log_unbin_spectra(self, flux=True, verbose=False):

      """
      Returns a un-logbinned version of the spectra

      Keywords
      --------
      flux:     Flag to conserve flux or not. Default: True
      verbose:  Flag for verbose
      
      Return
      ------
      Object instance with linearly binned spectra and updated info
    
      """

      logger.info("# Unbin ln spectra ...")

      if self.sampling == 'lin':
          logger.warning("Spectra already in linear lambda.")
          return copy(self)
          
      out      = copy(self)
      lamRange = [out.wave_init,out.wave_last]
      spec, wave  = cap.log_unbinning(lamRange, out.spec[:,0])
      out_spec = np.zeros((len(spec),out.nspec)) 
      for i in range(out.nspec): 
          out_spec[:,i], wave = cap.log_unbinning(lamRange, self.spec[:,i])
          if verbose:
             misc.printProgress(i+1,out.nspec)
     
      out.spec = out_spec
      out.wave = wave
      out.sampling = 'lin'
      out.update_basic_pars()
  
      return out

# -----------------------------------------------------------------------------
   def convolve_spectra(self, lsf_wave=None, lsf=None, mode='FWHM', verbose=False):

      """
      Returns a convolved version of the spectra

      Note
      ----
      this assumes SAMPLING='lin'
      If output LSF < input LSF setting bad values to input LSF

      Keywords
      --------
      lsf_wave:  Wavelength vector of output LSF
      lsf:       LSF vector
      mode:      FWHM/VDISP. First one in Angstroms. Second one in km/s
      verbose:   Flag for verbose
      
      Return
      ------
      Object instance with convolved spectra and updated info
    
      """

      logger.info("# Convolving spectra ...")

      out = copy(self)
      if mode == "FWHM":
         f_fwhm  = interpolate.interp1d(lsf_wave, lsf)
         out_lsf = f_fwhm(out.wave)
         in_lsf  = out.lsf_fwhm / 2.35  
         out.lsf_fwhm = out_lsf
      elif mode == "VDISP":
         f_vdisp = interpolate.interp1d(lsf_wave, lsf)
         out_lsf = f_vdisp(out.wave)
         in_lsf  = out.lsf_vdisp
         out.lsf_vdisp = out_lsf
      else:
         raise ValueError("Mode "+mode+" not a valid entry. Allowed values are FWHM/VDISP") 
  
      sigma    = np.sqrt(out_lsf**2 - in_lsf**2) / out.dwave
      bad = np.isnan(sigma)
      sigma[bad] = 1E-10
    
      out_spec = np.zeros_like(out.spec)     
      for i in range(out.nspec):
          out_spec[:,i] = cap.gaussian_filter1d(out.spec[:,i],sigma)
          if verbose:
             misc.printProgress(i+1,out.nspec, barLength=50)

      out.spec = out_spec
      
      return out  

# -----------------------------------------------------------------------------
   def tune_spectra(self, wave_lims=None, dwave=None, sampling=None, redshift=None,
                    lsf_flag=False, lsf_mode='FWHM',lsf_wave=None, lsf=None, verbose=False):

      """
      Returns the a tuned to desired input parameters

      Keywords
      --------
      wave_lims:     Wavelength limits in Angstroms
      dwave:         Step in wavelength (in Angstroms)
      sampling:      Type of sampling of the spectra. Valid inputs are lin/ln. Default: lin
      redshift:      Desired redshift
      lsf_flag:      Boolean flag to do LSF correction
      lsf_wave:      Wavelength vector of output LSF
      lsf:           LSF vector
      lsf_mode:      FWHM/VDISP. First one in Angstroms. Second one in km/s
      verbose:       Flag for verbose
      
      Return
      ------
      Object instance with tuned spectra and updated info
    
      """

      logger.info("# Tuning spectra ----------------------")

      out = copy(self)
      if lsf_wave is None:
          lsf_wave = out.wave

      # Resampling the spectra if necessary
      if (wave_lims[0] != out.wave_init) or (wave_lims[1] != out.wave_last) or (dwave != self.dwave):
         out = self.resample_spectra(wave_lims=wave_lims, dwave=dwave, verbose=verbose)

       # Redshift spectra if necessary
      if redshift != out.redshift:  
         out = out.redshift_spectra(redshift=redshift, verbose=verbose) 

      # Convolving spectra is necessary
      if lsf is not None:
         out = out.convolve_spectra(lsf_wave=lsf_wave, lsf=lsf, mode='FWHM', verbose=verbose) 

      # Log-rebinning spectra if necessary
      if sampling == 'ln':
         out = out.logrebin_spectra(verbose=verbose) 
          
      return out       

# -----------------------------------------------------------------------------
   def compute_save_mags(self, filters=None, zeropoint='AB', saveCSV=False, verbose=False):

      """
      Returns the magnitudes of the input spectra given a list of filters in a file

      Keywords
      --------
      filters:   Filters as provided by the method 'get_filters"
      zeropoint: Type of zero point. Valid inputs are AB/VEGA
      verbose:   Flag for verbose
      
      Return
      ------
      Dictionary with output magnitudes for each spectra for each filter
      If option saveCSV=True, returns .csv file with the magnitudes
    
      """
      logger.info("# Computing absolute magnitudes...")

      nfilters = len(filters.keys())
      zerosed  = utils.load_zerofile(zeropoint)   
      mags     = np.zeros((nfilters,self.nspec)) * np.nan   
      for i in range(self.nspec):
          mags[:,i] = utils.compute_mags(self.wave, self.spec[:,i], filters, zerosed, zeropoint)
          if verbose:
             misc.printProgress(i+1,self.nspec, barLength=50)

      outmags = {}
      for i in range(nfilters):
          outmags[list(filters.keys())[i]] = mags[i,:]

      if (saveCSV == True):
          logger.warning('Previous \'saved_mags.csv\' will be overwritten.')
          f=open('./pymiles/saved_files/saved_mags.csv', 'w') 
          writer = csv.writer(f)
          writer.writerows(outmags)
          f.close()

      return outmags

# -----------------------------------------------------------------------------
   def compute_ls_indices(self, saveCSV=False, verbose=False):

      """
      Returns the LS indices of the input spectra given a list of index definitions

      Keywords
      --------
      verbose: Flag for verbose
      
      Return
      ------
      Dictionary with output LS indices for each spectra for each index
      If option saveCSV=True, returns .csv file with the LS indices

      """
      logger.info("# Computing Line-Strength indices ...")

      # Getting the dimensions      
      names, indices, dummy = utils.lsindex(self.wave, self.spec[:,0], 0.0, self.redshift, 0.0, self.lsfile, plot=False, sims=0)
 
      nls     = len(indices)
      indices = np.zeros((nls,self.nspec))
    
      for i in range(self.nspec):
          names, indices[:,i], dummy = utils.lsindex(self.wave, self.spec[:,i], self.spec[:,i]*0.1, self.redshift, 0.0, self.lsfile)
          misc.printProgress(i+1,self.nspec)

      outls = {}     
      for i in range(nls):
          outls[names[i]] = indices[i,:]
 
      if (saveCSV == True):
          logger.warning('Previous \'saved_ls_indices.csv\' will be overwritten.')
          f=open('./pymiles/saved_files/saved_ls_indices.csv', 'w') 
          writer = csv.writer(f)
          writer.writerows(outls)
          f.close()
 
      return outls

# -----------------------------------------------------------------------------
   def save_object(self, filename, verbose=False):

      """
      Saves the contents of class instance into a HDF5 file 

      Arguments
      --------
      filename: Output filename (with full path)
      verbose:  Flag for verbose
      
      Return
      ------
      Nothing. File gets saved on disk
    
      """

      logger.info("# Saving object to "+filename)

      # Converting object to dictionary
      obj = self.__dict__

      # Saving contents in HDF5 file 
      if os.path.exists(filename):
           os.remove(filename)

      f = h5py.File(filename,"w")
      #------------------------------
      for key, value in obj.items():
          value = np.array(value)
          logger.debug(" - "+key, value.dtype.str)
          if value.dtype.str[1] == 'O':
              continue
          if np.ndim(obj[key]) == 1:
             if (value.dtype.str[1] == 'U'):
                value = str(value).encode("ascii","ignore")
             f.create_dataset(key, data=value)
          elif np.ndim(obj[key]) > 1:
             if (value.dtype.str[1] == 'U'):
               value = [n.encode("ascii","ignore") for n in value]
             f.create_dataset(key, data=value, compression="gzip")
      #------------------------------
      f.close()
    
      return

# -----------------------------------------------------------------------------
   @staticmethod
   def vacuum2air(wave_vac):

      """
      Converts wavelength from vacuum to air 

      Arguments
      --------
      None     

      Return
      ------
      Vector with wavelength in air system
    
      """

      wave_air = wave_vac / (1.0 + 2.735182E-4 + 131.4182 / wave_vac**2 + 2.76249E8 / wave_vac**4)

      return wave_air      

# -----------------------------------------------------------------------------
   def load_solar_spectrum(self):

      """
      Loads the references solar spectrum 

      Arguments
      --------
      None     

      Return
      ------
      Vector with wavelength in air system and flux
    
      """

      hdu = fits.open(self.solar_ref_spec)
      tab = hdu[1].data

      wave_air = self.vacuum2air(tab['WAVELENGTH'])
      flux     = tab['FLUX']

      return wave_air, flux      

# -----------------------------------------------------------------------------
   def compute_mag_sun(self, filters=None, zeropoint='AB', verbose=False):

      """
      Computes the magnitude of Sun in the desired filters 

      Arguments
      --------
      filters:   Filters as provided by the method 'get_filters"
      zeropoint: Type of zero point. Valid inputs are AB/VEGA
      verbose:   Flag for verbose

      Return
      ------
      Dictionary with solar mags for each filter
    
      """
      logger.info("# Computing solar absolute magnitudes...")

      wave, flux = self.load_solar_spectrum()
      nfilters   = len(filters.keys())
      zerosed    = utils.load_zerofile(zeropoint)   
      mags       = np.zeros((nfilters,self.nspec)) * np.nan   
      mags       = utils.compute_mags(wave, flux, filters, zerosed, zeropoint, sun=True)
   
      outmags = {}
      for i in range(nfilters):
          outmags[list(filters.keys())[i]] = mags[i]

      return outmags      

# -----------------------------------------------------------------------------
