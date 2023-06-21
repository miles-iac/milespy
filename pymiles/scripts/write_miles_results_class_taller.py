import os
import wget
import wave
import warnings
import numpy as np
import csv
from astropy.io import fits
from astropy.units.quantity_helper.function_helpers import where

#==============================================================================
class write_miles_results_class():

    warnings.filterwarnings("ignore")  

# -----------------------------------------------------------------------------
    def __init__(self, object=None):

        """
        Creates an instance of the class 
        
        Keywords
        --------
        wave:       Vector with input wavelengths in Angstroms
        spec:       [N,M] array with input spectra
        
        Return
        ------
        Object instance
        
        """
        
        if exist(object.imf_type):
        
          # Copying basic info
            self.wave      = np.array(wave)
            self.spec      = np.array(object.spec.shape[0])
            self.npix      = object.spec.shape[0]
            self.nspec     = object.spec.shape[1]
            self.age       = object.age
            self.met       = object.met
            self.alpha     = object.alpha 
            self.imf_slope = object.imf_slope
        #         self.filename  = None
            self.isochrone = object.isochrone
            self.imf_type  = object.imf_type
        #         self.index     = np.nan 
     

    
        return 
  
    def spec_to_fits(self):
 
        """
        Writes a given set of spectrum to .fits files 
        
        Arguments
        ---------
        wave:       Vector with input wavelengths in Angstroms
        spec:       [N,M] array with input spectra  
        
        Returns
        -------
        
        .fits files with the spectra
        
        """     
    for i in range(self.nspec):
        os.system('rm prueba.fits')
    
        # Create hdu
        hdu = fits.PrimaryHDU(data=self.spec[:,i])
        hdr = hdu.header
        hdr['NAXIS1'] = (self.npix,'\'  /  Length of axis')
        hdr['CRVAL1'] = (self.wave[0],'\' / central wavelength of first pixel')
        hdr['CDELT1'] = self.wave[1]-self.wave[0]
        
        if (self.source='MILES_SSP' or self.source='EMILES_SSP' or ):
    
            hdr['AGE']    = self.age
            hdr['MET']    = self.met
            hdr['alpha']  = self.alpha
            hdr['IMFshape'] = self.imf_shape
            hdr['IMFslope'] = self.imf_slope
  
            if (self.met<0):
                met_sign = 'm'
            if (self.met>=0):
                met_sign = 'p'
            if (self.alpha<0):
                alpha_sign = 'm'
            if (self.alpha>=0):
                alpha_sign = 'p'  
            filename = self.filename 
                            
#             filename = source[0]+imf_type+self.imf_slope+'Z'+met_sign++'T'+self.age+'_iT'+alpha_sign+self.alpha+'_'+

                 
            
            
#             hdr['BITPIX'] = ('8  /  8-bit ASCII characters')
#             hdr['NAXIS'] = ('1  /  Number of Image Dimensions')
#             hdr['NAXIS1'] = ('4300  /  Length of axis')
#             hdr['ORIGIN'] = ('\'NOAO-IRAF: WTEXTIMAGE\'  /')
#             hdr['IRAF-MAX'] = ('0.  /  Max image pixel (out of date)')
#             hdr['IRAF-MIN'] = ('0.  /  Min image pixel (out of date)')
#             hdr['IRAF-B/P'] = ('32  /  Image bits per pixel ')
#             hdr['IRAFTYPE'] = ('\'REAL FLOATING     \'  /  Image datatype')
#             hdr['OBJECT'] = ('\'A_3540.5-7409.6\'     /')
#             hdr['FILENAME'] = ('\'A.FITS          \'  /  IRAF filename')
#             hdr['FORMAT'] = ('\'5G14.7            \'  /  Text line format')
#             hdr['ORIGIN'] = ('\'NOAO-IRAF FITS Image Kernel July 1999\' / FITS file originator')
#             hdr['EXTEND'] = ('F / File may contain extensions')
#             hdr['DATE'] = ('\'21/06/09\'           / Date FITS file was generated')
#             hdr['IRAF-TLM'] = ('\'12:00:00 (21/06/09)\' / Time of last modification')
#             hdr['COMMENT'] = ('************************** REDUCEME HEADER ***************************')
#             hdr['HISTORY'] = ('From genimage')
#             hdr['CRVAL1'] = ('3540.5 / central wavelength of first pixel')
#             hdr['CDELT1'] = ('0.9000 / linear dispersion (Angstrom/pixel)')
#             hdr['FITSFI'] = ('\'[none]  \'           / Object name')
#             hdr['AIRMASS'] = ('0.00000 / Airmass')
#             hdr['TIMEXPO'] = ('0.0 / Timexpos')
#             hdr['WCSDIM'] = ('1')
#             hdr['CTYPE1'] = ('\'LINEAR  \'')
#             hdr['CD1_1'] = ('0.90')
#             hdr['LTM1_1'] = ('1.')
#             hdr['WAT0_001'] = ('\'system=equispectrum\'')
#             hdr['APNUM1'] = ('\'1 0 1.00 1.00 1.00 1.00\'')
#             hdr['DC-FLAG'] = ('0')
#             hdr['CRPIX1'] = ('1.')
#             hdr['WAT1_001'] = ('\'wtype=linear label=Wavelength units=angstroms\'')
            
            # Save it
            hdu.writeto(filename+'.fits')
        
        
        
        
        
        ### Writting .fits header
#         hdr = fits.Header.fromstring("""
#         BITPIX  =                    8  /  8-bit ASCII characters
#         NAXIS   =                    1  /  Number of Image Dimensions
#         NAXIS1  =                53688  /  Length of axis  
#         ORIGIN  = 'NOAO-IRAF: WTEXTIMAGE'  / 
#         IRAF-MAX=                   0.  /  Max image pixel (out of date)
#         IRAF-MIN=                   0.  /  Min image pixel (out of date)
#         IRAF-B/P=                   32  / Image bits per pixel
#         IRAFTYPE= \'REAL FLOATING     \'  /  Image datatype
#         OBJECT  = A_1680.20-49999.40     /
#         FILENAME= E.FITS            /  IRAF filename
#         FORMAT  = \'5G14.7            \'  /  Text line format
#         ORIGIN = \'NOAO-IRAF FITS Image Kernel July 1999\' / FITS file originator
#         EXTEND  =                    F / File may contain extensions
#         DATE    = 01/03/16          / Date FITS file
#         IRAF-TLM= 10:41:25 (05/03/20) / Last modification 
#         COMMENT   ************************** REDUCEME HEADER ***************************
#         HISTORY   From genimage
#         CRVAL1  =             1680.20 / central wavel pixel 1
#         CDELT1  =              0.9000 / linear dispersion (Angstrom/pixel)
#         FITSFI  = \'[none]  \'           / Object name
#         AIRMASS =              0.00000 / Airmass
#         TIMEXPO =                  0.0 / Timexpos
#         WCSDIM = 1
#         CTYPE = \'LINEAR  \'
#         CD1_1   =                 0.90
#         LTM1_1  =                   1.
#         WAT0_001= \'system=equispec\'
#         APNUM1  = \'1 0 1.00 1.00 1.00 1.00\'
#         DC-FLAG =                    0
#         CRPIX1  =                   1. 
#         WAT1_001= \'wtype=linear label=Wavelength units=angstroms\'
#         END
#         """)
        
# hdr = fits.Header.fromstring("""
#         BITPIX  =                    8  /  8-bit ASCII characters \n                   
#         NAXIS   =                    1  /  Number of Image Dimensions \n                  
#         NAXIS1  =                53688  /  Length of axis \n                              
#         ORIGIN  = 'NOAO-IRAF: WTEXTIMAGE'  / \n                                           
#         IRAF-MAX=                   0.  /  Max image pixel (out of date) \n               
#         IRAF-MIN=                   0.  /  Min image pixel (out of date) \n
#         IRAF-B/P=                   32  / Image bits per pixel \n
#         IRAFTYPE= \'REAL FLOATING     \'  /  Image datatype \n                              
#         OBJECT  = A_1680.20-49999.40     / \n                                             
#         FILENAME= E.FITS            /  IRAF filename \n                                   
#         FORMAT  = \'5G14.7            \'  /  Text line format \n  
#         ORIGIN = \'NOAO-IRAF FITS Image Kernel July 1999\' / FITS file originator \n  
#         EXTEND  =                    F / File may contain extensions \n                   
#         DATE    = 01/03/16          / Date FITS file \n
#         IRAF-TLM= 10:41:25 (05/03/20) / Last modification \n  
#         COMMENT   ************************** REDUCEME HEADER ***************************\n
#         HISTORY   From genimage \n                                                        
#         CRVAL1  =             1680.20 / central wavel pixel 1 \n 
#         CDELT1  =              0.9000 / linear dispersion (Angstrom/pixel) \n            
#         FITSFI  = \'[none]  \'           / Object name \n                                   
#         AIRMASS =              0.00000 / Airmass \n    
#         TIMEXPO =                  0.0 / Timexpos \n
#         WCSDIM = 1 \n
#         CTYPE = \'LINEAR  \' \n
#         CD1_1   =                 0.90 \n                                                 
#         LTM1_1  =                   1. \n                                                 
#         WAT0_001= \'system=equispec\' \n                
#         APNUM1  = \'1 0 1.00 1.00 1.00 1.00\' \n
#         DC-FLAG =                    0 \n
#         CRPIX1  =                   1. \n                                                 
#         WAT1_001= \'wtype=linear label=Wavelength units=angstroms\' \n                      
#         END \n
#         """)
#         hdu = fits.PrimaryHDU()
                                                  
#         
        ### Writting .fits data: wave and spectrum
        data = np.zeros((len(self.wave),2))
        data[:,0] = self.wave
        for i in range(self.nspec):
            os.system('rm prueba.fits')
            data[:,1] = self.spec[:,i]
            
            hdu = fits.PrimaryHDU()

            hdu.data = data
#             hdu.header['BITPIX'] = ('8  /  8-bit ASCII characters')
#             hdu.header['NAXIS'] = ('1  /  Number of Image Dimensions')
            hdu.header['NAXIS1'] = ('4300  /  Length of axis')
            hdu.header['ORIGIN'] = ('\'NOAO-IRAF: WTEXTIMAGE\'  /')
            hdu.header['IRAF-MAX'] = ('0.  /  Max image pixel (out of date)')
            hdu.header['IRAF-MIN'] = ('0.  /  Min image pixel (out of date)')
            hdu.header['IRAF-B/P'] = ('32  /  Image bits per pixel ')
            hdu.header['IRAFTYPE'] = ('\'REAL FLOATING     \'  /  Image datatype')
            hdu.header['OBJECT'] = ('\'A_3540.5-7409.6\'     /')
            hdu.header['FILENAME'] = ('\'A.FITS          \'  /  IRAF filename')
            hdu.header['FORMAT'] = ('\'5G14.7            \'  /  Text line format')
            hdu.header['ORIGIN'] = ('\'NOAO-IRAF FITS Image Kernel July 1999\' / FITS file originator')
#             hdu.header['EXTEND'] = ('F / File may contain extensions')
            hdu.header['DATE'] = ('\'21/06/09\'           / Date FITS file was generated')
            hdu.header['IRAF-TLM'] = ('\'12:00:00 (21/06/09)\' / Time of last modification')
            hdu.header['COMMENT'] = ('************************** REDUCEME HEADER ***************************')
            hdu.header['HISTORY'] = ('From genimage')
            hdu.header['CRVAL1'] = ('3540.5 / central wavelength of first pixel')
            hdu.header['CDELT1'] = ('0.9000 / linear dispersion (Angstrom/pixel)')
            hdu.header['FITSFI'] = ('\'[none]  \'           / Object name')
            hdu.header['AIRMASS'] = ('0.00000 / Airmass')
            hdu.header['TIMEXPO'] = ('0.0 / Timexpos')
            hdu.header['WCSDIM'] = ('1')
            hdu.header['CTYPE1'] = ('\'LINEAR  \'')
            hdu.header['CD1_1'] = ('0.90')
            hdu.header['LTM1_1'] = ('1.')
            hdu.header['WAT0_001'] = ('\'system=equispectrum\'')
            hdu.header['APNUM1'] = ('\'1 0 1.00 1.00 1.00 1.00\'')
            hdu.header['DC-FLAG'] = ('0')
            hdu.header['CRPIX1'] = ('1.')
            hdu.header['WAT1_001'] = ('\'wtype=linear label=Wavelength units=angstroms\'')
            
            if (SSP):
              Mku1.30Zm0.25T00.0500_iTp0.00_baseFe
                filename = source[0]+imf_type+??+'Z'+m/p??+'T'+??+'_iT'+p/m??+'_'+??
            if (CSP):
                filename =
            if (star):
                filename = ''
            if (SFH):
                filename = 
            hdu.writeto('prueba.fits')
            
            """
            hdu = fits.PrimaryHDU(data=data, hdr=hdr)
            hdr = hdu.header
            hdr['BITPIX'] = ('8  /  8-bit ASCII characters')
            hdr['NAXIS'] = ('1  /  Number of Image Dimensions')
            hdr['NAXIS1'] = ('4300  /  Length of axis')
            hdr['ORIGIN'] = ('\'NOAO-IRAF: WTEXTIMAGE\'  /')
            hdr['IRAF-MAX'] = ('0.  /  Max image pixel (out of date)')
            hdr['IRAF-MIN'] = ('0.  /  Min image pixel (out of date)')
            hdr['IRAF-B/P'] = ('32  /  Image bits per pixel ')
            hdr['IRAFTYPE'] = ('\'REAL FLOATING     \'  /  Image datatype')
            hdr['OBJECT'] = ('\'A_3540.5-7409.6\'     /')
            hdr['FILENAME'] = ('\'A.FITS          \'  /  IRAF filename')
            hdr['FORMAT'] = ('\'5G14.7            \'  /  Text line format')
            hdr['ORIGIN'] = ('\'NOAO-IRAF FITS Image Kernel July 1999\' / FITS file originator')
            hdr['EXTEND'] = ('F / File may contain extensions')
            hdr['DATE'] = ('\'21/06/09\'           / Date FITS file was generated')
            hdr['IRAF-TLM'] = ('\'12:00:00 (21/06/09)\' / Time of last modification')
            hdr['COMMENT'] = ('************************** REDUCEME HEADER ***************************')
            hdr['HISTORY'] = ('From genimage')
            hdr['CRVAL1'] = ('3540.5 / central wavelength of first pixel')
            hdr['CDELT1'] = ('0.9000 / linear dispersion (Angstrom/pixel)')
            hdr['FITSFI'] = ('\'[none]  \'           / Object name')
            hdr['AIRMASS'] = ('0.00000 / Airmass')
            hdr['TIMEXPO'] = ('0.0 / Timexpos')
            hdr['WCSDIM'] = ('1')
            hdr['CTYPE1'] = ('\'LINEAR  \'')
            hdr['CD1_1'] = ('0.90')
            hdr['LTM1_1'] = ('1.')
            hdr['WAT0_001'] = ('\'system=equispectrum\'')
            hdr['APNUM1'] = ('\'1 0 1.00 1.00 1.00 1.00\'')
            hdr['DC-FLAG'] = ('0')
            hdr['CRPIX1'] = ('1.')
            hdr['WAT1_001'] = ('\'wtype=linear label=Wavelength units=angstroms\'')
            hdr.tofile('prueba.fits')
            hdu.writeto('prueba.fits')
            """
            
        return
#     
#     def upload_spec_results(self):    
#     
#         outdir  = "../spec_results/"
#         baseurl = "http://127.0.0.1:5000/download/"
#         n_mag = "miles_ivr.MAG"
#         n_ind = "miles_ivr.???"
#         n_png = "miles_ivr.png"
#         n_tgz = "miles_ivr.tgz"
# #     
#         nlist = [n_mag, n_ind, n_png, n_tgz]
#         # Downloading files if needed
#         for i in range(nlist):
#             url     = baseurl+nlist[i]
#             outname = outdir+nlist[i]
#             filename = wget.upload(url,out=outname, bar=None)
#             os.remove(outname)     



Preguntar:
    - Nombre ficheros: clase aparte o funcion nueva: clase aparte
    - SBF function -> SSP encontrada o interpolada?: mas tarde
    - CSP -> modulo SFH: mas tarde
    - Header mas corto o igual: ver espectros de miles actuales
    - Mku1.30Zm0.25T00.0500_iTp0.00isocrona_baseFe
    
    def dual_CSP(self, age):
    
    
    get_ssp_in_list(self, age_list=None, met_list=None, alpha_list=None, 
                   imf_slope_list=None, verbose=False):
    
  def compute_SBF(self, ):
      get_ssp_by_params(self, age=None, met=None, alpha=None, imf_slope=None, return_pars=False, verbose=False):
