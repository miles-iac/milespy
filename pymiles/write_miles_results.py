import os
import warnings
import numpy as np
from astropy.io import fits
from fileinput import filename

# ==============================================================================


class write_miles_results():

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
        self.source = object.source
        self.version = object.version
        self.wave = np.array(object.wave)
        self.npix = len(object.wave)
        self.spec = np.array(object.spec)
        self.nspec = object.nspec
        self.filename = object.filename

        print('object.time' in locals())
        # or SSP Models Class
        if ((object.source == 'MILES_SSP' or object.source == 'EMILES_SSP' or object.source == 'CaT_SSP') and ('object.time' not in locals())):
          # Copying basic info
            #         self.index     = np.nan
            self.imf_type = object.imf_type
            print(self.imf_type)
            self.imf_slope = object.imf_slope
            self.isochrone = object.isochrone
            self.age = object.age
            self.met = object.met
            self.alpha = object.alpha
            self.class_flag = 'SSP'

        # For Stellar Library Class
        if (object.source == 'MILES_STARS' or object.source == 'EMILES_STARS' or object.source == 'CaT_STARS'):
          # Copying basic info
            self.index = object.index
            self.starname = object.starname
            self.id = object.id
            self.teff = object.teff
            self.logg = object.logg
            self.FeH = object.FeH
            self.MgFe = object.MgFe
            self.class_flag = 'STARS'

        if ((object.source == 'MILES_SSP' or object.source == 'EMILES_SSP' or object.source == 'CaT_SSP') and ('object.time' in locals())):
            # Copying basic info
            self.time = object.time
            self.wts = object.wts
            self.sfr = object.sfr
            self.alpha_evol = object.alpha_evol
            self.imf_evol = object.imf_evol
            self.met_evol = object.met_evol
            self.class_flag = 'SFH'

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
            filename = self.filename[i]
            os.system('rm ' + filename + '.fits')

            # Create hdu
            hdu = fits.PrimaryHDU(data=self.spec[:, i])
            hdr = hdu.header

            # Create header basic header
            hdr['NAXIS1'] = self.npix, '\'  /  Length of axis'
            hdr['COMMENT'] = 'FITS (Flexible Image Transport System) format is defined in \'Astronomy'
            hdr['COMMENT'] = 'and Astrophysics\', volume 376, page 359; bibcode: 2001A&A...376..359H'
            hdr['CRVAL1'] = self.wave[0], '\' / central wavelength of first pixel'
            hdr['CDELT1'] = self.wave[1] - self.wave[0]
            hdr['CRPIX1'] = '1'

            # Copyright
            hdr['COMMENT'] = '#==============================================================='
            hdr['COMMENT'] = '# Copyright (C) 2016, MILES team'
            hdr['COMMENT'] = '# Email: miles@iac.es'
            hdr['COMMENT'] = '#       '
            hdr['COMMENT'] = '# This spectrum has been produced using the Webtool facilities'
            hdr['COMMENT'] = '# of the MILES website (http://miles.iac.es) hosted at the'
            hdr['COMMENT'] = '# Instituto de Astrofisica de Canarias using the population'
            hdr['COMMENT'] = '# synthesis code "MILES" (MIL EStrellas). See Vazdekis (1999),'
            hdr['COMMENT'] = '# Vazdekis et al. (2003, 2010, 2015, 2016) for details on the code'
            hdr['COMMENT'] = '# implementation.'
            hdr['COMMENT'] = '#       '
            hdr['COMMENT'] = '#                   --------------------'
            hdr['COMMENT'] = '#       '
            hdr['COMMENT'] = '# If you have found this website and its products useful'
            hdr['COMMENT'] = '# for your research, please refer to the data as obtained'
            hdr['COMMENT'] = '# "from the MILES website", while we suggest to add to your'
            hdr['COMMENT'] = '# acknowledgements:'
            hdr['COMMENT'] = '#       '
            hdr['COMMENT'] = '#     "This research is/was (partially) based on data'
            hdr['COMMENT'] = '#                 from the MILES project."'
            hdr['COMMENT'] = '#       '
            hdr['COMMENT'] = '# We would appreciate receiving a preprint for our records.'
            hdr['COMMENT'] = '#       '
            hdr['COMMENT'] = '#                   --------------------'
            hdr['COMMENT'] = '#       '
            hdr['COMMENT'] = '# This spectrum is provided as is without any warranty whatsoever.'
            hdr['COMMENT'] = '# Permission to use, for non-commercial purposes is granted.'
            hdr['COMMENT'] = '# Permission to modify for personal or internal use is granted,'
            hdr['COMMENT'] = '# provided this copyright and disclaimer are included unchanged'
            hdr['COMMENT'] = '# at the beginning of the file. All other rights are reserved.'
            hdr['COMMENT'] = '#==============================================================='

            # Create header for specific spectra
            # Header for SSP spectra
            if (self.class_flag == 'SSP'):
                hdr['COMMENT'] = '#       '
                hdr['COMMENT'] = '# WEBTOOL: Tune SSP Models'
                hdr['COMMENT'] = '#       '
                hdr['COMMENT'] = '# MODEL INGREDIENTS --------------------------------------------'
                hdr['COMMENT'] = '#       '
                hdr['MODELS LIBRARY'] = self.source
                hdr['VERSION'] = self.version
                hdr['IMF, SLOPE'] = self.imf_type[i], self.imf_slope[i]
                hdr['AGE [GYR]'] = self.age[i]
                hdr['[M/H]'] = self.met[i]
                hdr['[alpha/Fe]'] = str(self.alpha[i])
                hdr['ISOCHRONE'] = self.isochrone[i]
#                 hdr['COMMENT'] = '#       '
#                 hdr['COMMENT'] = '# SPECTRUM INFORMATION -----------------------------------------'
#                 hdr['COMMENT'] = '#       '
#                 hdr['SAMPLING'] = self.sampling
#                 hdr['RESOLUTION, FWHM (A)'] = self.lsf_fwhm
#                 hdr['RESOLUTION, VEL. DISP (km/s)'] = self.lsf_vdisp
#                 hdr['REDSHIFT'] = self.redshift
                hdr['COMMENT'] = '#       '
                hdr['COMMENT'] = '#==============================================================='

            # Header for SSP spectra
            if (self.class_flag == 'STARS'):
                hdr['COMMENT'] = '#       '
                hdr['COMMENT'] = '# WEBTOOL: Spectra by Stellar Parameters'
                hdr['COMMENT'] = '#       '
                hdr['COMMENT'] = '# MODEL INGREDIENTS --------------------------------------------'
                hdr['COMMENT'] = '#       '
                hdr['MODELS LIBRARY'] = self.source
                hdr['VERSION'] = self.version
                hdr['STAR NAMES'] = self.starname
#                 hdr['COMMENT'] = '# List of stars for the interpolation:'
#                 hdr['COMMENT'] = '# Star name, Spectrum, Teff, Logg, [Fe/H], [Mg/Fe]'
#                 hdr['COMMENT'] = '       # XXXXXXXXX   XXXXX    XXXX  XXXX   XXXX    XXXX
#                 hdr['COMMENT'] = '       # XXXXXXXXX   XXXXX    XXXX  XXXX   XXXX    XXXX
#                 COMMENT = '#       '
#                 COMMENT = '# Requested Parameters: XXXXX, XX, XX'
#                 COMMENT = '# Output Parameters: XXXXX,XX,XX'
                hdr['TEMP EFF [K]'] = self.teff[i]
                hdr['LOG G'] = self.logg[i]
                hdr['[Fe/H]'] = self.FeH[i]
                hdr['[Mg/Fe]'] = self.MgFe[i]

                hdr['COMMENT'] = '#       '
                hdr['COMMENT'] = '# SPECTRUM INFORMATION -----------------------------------------'
                hdr['COMMENT'] = '#       '
                hdr['SAMPLING'] = self.sampling
                hdr['RESOLUTION, FWHM (A)'] = self.lsf_fwhm
                hdr['RESOLUTION, VEL. DISP (km/s)'] = self.lsf_vdisp
                hdr['REDSHIFT'] = self.redshift
                hdr['COMMENT'] = '#       '
                hdr['COMMENT'] = '#==============================================================='

#             if ( self.class_flag == 'SFH' ):

            # Save it
            hdu.writeto(filename + '.fits')


#
# - He modificado SSP_class. He hecho listas y añadido el filename
# - He separado routa y nombre
# - He descubierto que no funcionan bien dos funciones
#
# - He modificado stellar_lib_class. He añadido create_new_object por simetria.
# - Hay una linea restante por modificiar.
# - Que pasa con el nombre de las estrellas que es 's0180.fits'
#
# - En write_miles_class no sé como y cuando guardar el redshift y demas
# - Tampoco sé muy bien cual es la forma mas optima de pasarle en algunos casos age, met... y otros teff,loggg..
