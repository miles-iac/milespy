import os
import sys
import glob
import logging
import h5py
import numpy                 as np
import matplotlib.pyplot     as plt
import pymiles.scripts.misc_functions        as misc
import pymiles.scripts.pymiles_utils        as utils
from   astropy.io            import ascii, fits
from   scipy.spatial         import Delaunay
from   copy                  import copy
from   pymiles.scripts.tuning_tools import tuning_tools
#==============================================================================

logger = logging.getLogger('pymiles.lib')

class stellar_library(tuning_tools):

# -----------------------------------------------------------------------------
   def __init__(self, source='MILES_STARS', version='9.1'):

       """
       Creates an instance of the class 
       
       Keywords
       --------
       source:  Name of input models to use. Valid inputs are MILES_STARS/CaT_STARS/EMILES_STARS
       version: Version number of the models
    
       Return
       ------
       Object instance
    
       """
       repo_filename = "./pymiles/repository/"+source+"_v"+version+".hdf5"

       # Opening the relevant file in the repository
       f = h5py.File(repo_filename,"r")
       #------------------------------
       self.index    = np.array(f['index'])
       self.teff     = np.array(f['teff'])
       self.logg     = np.array(f['logg'])
       self.FeH      = np.array(f['FeH'])
       self.MgFe     = np.array(f['MgFe'])
       self.starname = [n.decode() for n in f['starname']]
       self.filename = [n.decode() for n in f['filename']]
       self.id       = [np.int(n.decode()) for n in f['id']]
       self.nspec    = np.amax(self.index+1)
       self.wave     = np.array(f['wave'])
       self.spec     = np.array(f['spec'])
       self.source   = source
       self.version  = version
       #------------------------------
       f.close()

       # Flagging if all elements of MgFe are NaNs
       self.MgFe_flag = 0     
       if (np.nansum(self.MgFe) == 0):
          self.MgFe_flag = 1

       # Creating Delaunay triangulation of parameters for future searches and interpolations
       if self.MgFe_flag == 1:
          idx    = np.isfinite(self.teff) & np.isfinite(self.logg) & np.isfinite(self.FeH)
          ngood  = np.sum(idx)
          self.params = np.empty((ngood,3))
          self.params[:,0] = np.log10(self.teff)[idx]
          self.params[:,1] = self.logg[idx]
          self.params[:,2] = self.FeH[idx]
       else:
          idx    = np.isfinite(self.teff) & np.isfinite(self.logg) & np.isfinite(self.FeH) & np.isfinite(self.MgFe)
          ngood  = np.sum(idx)
          self.params = np.empty((ngood,4))
          self.params[:,0] = np.log10(self.teff)[idx]
          self.params[:,1] = self.logg[idx]
          self.params[:,2] = self.FeH[idx]
          self.params[:,3] = self.MgFe[idx]

       self.tri       = Delaunay(self.params)
       self.new_index = self.index[idx]
       self.main_keys = list(self.__dict__.keys())

       # Inheriting the tuning_tools class
       super().__init__(source=self.source,wave=self.wave,spec=self.spec)

# -----------------------------------------------------------------------------
   def set_item(self,idx):

       """
       Creates a copy of input instance and slices the arrays for input indices

       Arguments
       ---------
       idx: integer or boolean array indicating the elements to be extracted

       Returns
       -------
       Object instance for selected items

       """

       out = copy(self)
       out.index    = np.array(self.index)[idx]
       out.teff     = np.array(self.teff)[idx]
       out.logg     = np.array(self.logg)[idx]
       out.FeH      = np.array(self.FeH)[idx]
       out.MgFe     = np.array(self.MgFe)[idx]
       out.starname = np.array(self.starname)[idx]
       out.filename = np.array(self.filename)[idx]
       out.id       = np.array(self.id)[idx]
       out.nspec    = 1
       out.wave     = np.array(self.wave)
       out.spec     = np.array(self.spec)[:,idx]

       return out

# -----------------------------------------------------------------------------
   def search_by_id(self,id=None):

       """
       Searches a star in database for a given ID

       Arguments
       ---------
       id: integer with the star ID in database

       Returns
       -------
       Object instance for selected items

       """

       idx = (np.array(self.id) == id)
       if (np.sum(idx) == 0):
           raise ValueError("No star with that ID")
       
       out = self.set_item(idx)

       return out

 # -----------------------------------------------------------------------------
   def get_starname(self,id=None):

       """
       Gets a starname in database for a given ID

       Arguments
       ---------
       id: integer with the star ID in database

       Returns
       -------
       Star name

       """

       idx = (np.array(self.id) == id)
       if (np.sum(idx) == 0):
           raise ValueError("No star with that ID")
       
       out = self.set_item(idx)
       
       return out.starname
 # -----------------------------------------------------------------------------
   def get_stars_in_range(self, teff_lims=None, logg_lims=None, FeH_lims=None, MgFe_lims=None):

       """
       Gets set of stars with parameters range

       Arguments
       ---------
       teff_lims: Limits in Teff
       logg_lims: Limits in Log(g)
       FeH_lims:  Limits in [Fe/H]
       MgFe_lims: Limits in [Mg/Fe]

       Returns
       -------
       Object instance for stars within parameters range

       """

       if (self.MgFe_flag == 1):
          idx = (self.teff >= teff_lims[0]) & (self.teff <= teff_lims[1]) & \
                (self.logg >= logg_lims[0]) & (self.logg <= logg_lims[1]) & \
                (self.FeH  >= FeH_lims[0])  & (self.FeH  <= FeH_lims[1]) 
       else:         
          idx = (self.teff >= teff_lims[0]) & (self.teff <= teff_lims[1]) & \
                (self.logg >= logg_lims[0]) & (self.logg <= logg_lims[1]) & \
                (self.FeH  >= FeH_lims[0])  & (self.FeH  <= FeH_lims[1])  & \
                (self.MgFe >= MgFe_lims[0]) & (self.MgFe <= MgFe_lims[1])

       out = self.set_item(idx)

       return out

# -----------------------------------------------------------------------------
   def search_by_params(self, teff=None, logg=None, FeH=None, MgFe=None):

       """
       Gets closest star in database for given set of parameters

       Arguments
       ---------
       teff: Desired Teff
       logg: Desired Log(g)
       FeH:  Desired [Fe/H]
       MgFe: Desired [Mg/Fe]

       Returns
       -------
       Object instance for closest star

       """

       # Searching for the simplex that surrounds the desired point in parameter space
       if self.MgFe_flag == 1:
          input_pt = np.array([np.log10(teff),logg,FeH], ndmin=2)
       else:
          input_pt = np.array([np.log10(teff),logg,FeH,MgFe], ndmin=2)

       vtx, wts = utils.interp_weights(self.params, input_pt, self.tri)
       vtx, wts = vtx.ravel(), wts.ravel()
       
       # Deciding on the closest vertex and extracting info
       idx     = np.argmax(wts)
       new_idx = self.new_index[vtx[idx]]
       out     = self.set_item(new_idx)

       return out

# -----------------------------------------------------------------------------
   def get_spectrum_by_params_delaunay(self, teff=None, logg=None, FeH=None, MgFe=None):

       """
       Interpolates a star spectrum for given set of parameters using Delaunay triangulation

       Arguments
       ---------
       teff: Desired Teff
       logg: Desired Log(g)
       FeH:  Desired [Fe/H]
       MgFe: Desired [Mg/Fe]

       Returns
       -------
       wave: wavelength of output spectrum
       spec: interpolated spectrum

       """

      # Searching for the simplex that surrounds the desired point in parameter space
       if self.MgFe_flag == 1:
          input_pt = np.array([np.log10(teff),logg,FeH], ndmin=2)
       else:
          input_pt = np.array([np.log10(teff),logg,FeH,MgFe], ndmin=2)

       vtx, wts = utils.interp_weights(self.params, input_pt, self.tri)
       vtx, wts = vtx.ravel(), wts.ravel()
 
       new_idx = self.new_index[vtx]

       wave = self.wave
       spec = np.dot(self.spec[:,new_idx],wts)
#        print(wts)
       
       # Saving all the new info into an object 
       out = self.create_new_object(teff, logg, FeH, MgFe, wave, spec, vtx, wts)

       return out

# -----------------------------------------------------------------------------
# CREATE_NEW_OBJECT
#
# Creates a new object from an interpolated spectra in get_spectrum_by_params_delaunay
# -----------------------------------------------------------------------------
   def create_new_object(self, teff, logg, FeH, MgFe, wave, spec, indices, weights):
  
      """
      Creates a new object using the info from the get_spectrum_by_params_delaunay method 

      Arguments
      --------
      teff:      Desired Teff
      logg:      Desired Log(g)
      FeH:       Desired [Fe/H]
      MgFe:      Desired [Mg/Fe]
      wave:      Input wavelength
      spec:      Interpolated spectrum
      indices:   Elements of the original object to do the interpolation
      weights:   Weights for each of the elements

      Return
      ------
      Dictionary with mass-to-light ratios for each SSP model and filter
    
      """

      # Copying basic info
      out           = copy(self)
      nspec_in      = self.nspec
      out.wave      = wave
      out.spec      = np.array(spec,ndmin=2).T
      out.teff      = [teff]
      out.logg      = [logg]
      out.FeH       = [FeH] 
      out.MgFe      = [MgFe]
      out.nspec     = 1
      out.index     = np.nan 
      
      ### Creating filenames 
      if (teff < 10000.):
          teff_str = 'T0%.1f' %teff
      if (teff >= 10000.):
          teff_str = 'T%.1f' %teff
      out.filename = out.source+'_Teff'+teff_str+'_Logg%.2f_FeH%.2f_MgFe%.2f' %(logg, FeH, MgFe)
      # Interpolating other parameters
      # NOTE: The problem with this is that we are hard-coding a few attributes
      #       but it is probably not too bad. We hard code only minimum necessary
      keys = list(out.main_keys)
      for i in range(len(out.main_keys)):
         if (keys[i] == 'wave') or (keys[i] == 'spec') or (keys[i] == 'nspec') or \
            (keys[i] == 'teff')  or (keys[i] == 'logg') or (keys[i] == 'FeH') or  \
            (keys[i] == 'MgFe'):
            continue
         val = np.array(getattr(out,keys[i]))
         
         if (np.ndim(val) == 0):
            continue
         if (val.shape[0] == nspec_in):
            print('VER ESTA PARTE')
#             setattr(out, keys[i], np.dot(val[indices],weights))         
      
      # Instaitiating the tuning_tools class
#       super().__init__(source=out.source,wave=out.wave,spec=out.spec) ???

      return out

   
