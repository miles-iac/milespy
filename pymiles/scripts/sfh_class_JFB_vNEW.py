import os
import sys
import glob
import h5py
import pymiles_utils       as utils
import numpy               as np
import matplotlib.pyplot   as plt
import misc_functions      as misc
from   astropy.io          import ascii, fits
from   scipy.spatial       import Delaunay
from   scipy.integrate     import simps
from   copy                import copy
#==============================================================================
class sfh():

# -----------------------------------------------------------------------------
    def __init__(self):

       """
       Creates an instance of the class 
       
       Keywords
       --------
       None
    
       Return
       ------
       Object instance
    
       """
    
       # The SFH is defined based on a time array
       self.time       = np.sort(list(set(self.age)))
       self.wts        = np.zeros_like(self.time)
       self.sfr        = np.zeros_like(self.time)
       self.alpha_evol = np.zeros_like(self.time) 
       self.imf_evol   = np.zeros_like(self.time) + 1.3
       self.met_evol   = np.zeros_like(self.time)

    # ----------
    # Methods to define the time array. This is the fundation of 
    # the SFHs as it defines where the SSPs will be evalated
    # ----------
    def set_time_range(self, start=14., end=0.):
       """Sets the beginning and the end of the SFH
       
       Defines the start (oldest) and the end (youngest) of the 
       ouput SFH based on the limits set by the user and the 
       grid provided by the isochrones.
       
       Parameters
       ----------

       start : scalar (Gyr)
            Defines the beggining of the SFH. Only SSPs younger 
            than start are included (default=14)
       end   : scalar (Gyr)
            Defined the end of the SFH. Only SSPs older than 
            end are included (default=0)

       Returns
       -------

       Sets the age array over which the SFH will be constructed and
       update the rest of the variables accordingly
       """
       
       assert ( np.isscalar(start) and np.isscalar(end) ), \
                "Start and end times must be scalar" 
       assert (start > end), \
                "Start date must be older than end date" 


       idx = (self.age <= start) & (self.age >= end) 
       self.time = np.sort(list(set(self.age[idx])))
       self.met_evol   = np.zeros_like(self.time)
       self.alpha_evol = np.zeros_like(self.time) 
       self.imf_evol   = np.zeros_like(self.time) + 1.3
    
    def set_time_user(self, user_time):
       """User-defined age array
       
       Sets the input array as the time axis for the SFH
       
       Parameters
       ----------

       user_time : array (Gyr)
            User-defined array of ages to evaluate the SFH

       Returns
       -------

       Sets the age array over which the SFH will be constructed and
       update the rest of the variables accordingly
       """  
       assert (len(user_time) >= 2), \
            "You should not be here... only one age was provided.  Look for SSP models instead." 
       
       self.time       = np.sort(list(set(user_time)))
       self.met_evol   = np.zeros_like(self.time)
       self.alpha_evol = np.zeros_like(self.time) 
       self.imf_evol   = np.zeros_like(self.time) + 1.3

    # ----------
    # Methods to define the actual SFR as a function of time
    # Although SSPs are combined according to their contribution
    # to the mass budget, most methods define the SFR instead. Weights
    # in mass are calculated as weight(t) = SFR(t) * dt where "dt" is 
    # the time increment between two consecutive age bins. 
    #
    # Note on the normalization. The SFR is normalized so it forms one 
    # solar mass over the duration of the burst. Weights are normalized 
    # in the same way.
    # ----------
    def constant_sfr(self,start=14,end=1.,met=0.,alpha=0,imf_slope=1.3):
        """SFH with a constant SFR

        Defines a constant SFR over a period of time defined by the user.
        Metallicities, [alpha/Fe], and IMF slopes are kept constant over 
        the whole burst (although they can be updated with the additional
        methods).

       Parameters
       ----------

       start : scalar (Gyr)
            Start of the burst (default=14)
       end   : scalar (Gyr)
            End of the burst (default=1)
       met   : scalar (dex)
            (log) Metallicity of newly formed stars (default=0)
       alpha : scalar (dex)
            (log) [alpha/Fe] of newly formed stars (default=0)
       imf_slope : scalar 
            logarithmic IMF slope of newly formed stars (default=1.3)

       Returns
       -------

       Updates the SFH parameters of the instance
       """  

        assert (
            np.isscalar(start) and np.isscalar(end) and np.isscalar(met) and np.isscalar(alpha) and np.isscalar(imf_slope) 
            ), \
            "Input parameters must be scalar" 

        assert (start >= end), "Start time must happen befor the end of the burst"

        # Constant SFR
        self.sfr  = np.zeros_like(self.time)
        self.sfr[(self.time <= start) & (self.time >= end)] = 1
        # Normalization to form 1 Msun
        norm = simps(self.sfr,self.time)
        self.sfr  = self.sfr / norm
        # Mass-weights
        dt = (self.time-np.roll(self.time,1))
        dt[0] = dt[1]
        self.wts  = self.sfr * dt
        # Normalized to a total of 1 Msun
        self.wts  = self.wts / np.sum(self.wts)

        # Set as constant the other SFH parameters 
        self.met_evol   = np.zeros_like(self.time) + met
        self.alpha_evol = np.zeros_like(self.time) + alpha
        self.imf_evol = np.zeros_like(self.time) + imf_slope

        # Make sure everything is within the allowed range of models
        assert (self.sfh_check_ssp_range() ),\
          "Input SFH params are outside the MILES model grid you loaded"

    def tau_sfr(self,start=10, tau=1, met=0, alpha=0, imf_slope=1.3):
        """Exponentially declinging SFR

        This is a standard tau model where the SFR is given by
            SFR(t) = 0 for t < start
            SFR(t) = exp (- (t-start)/tau)) for t >= start
        
        The normalization is such one solar mass is formed

        Parameters
        ----------

        start : scalar (Gyr)
            Start of the burst (default=14)
        tau   : scalar (Gyr)
            e-folding time (default=1)
        met   : scalar (dex)
            (log) Metallicity of newly formed stars (default=0)
        alpha : scalar (dex)
            (log) [alpha/Fe] of newly formed stars (default=0)
        imf_slope : scalar 
            logarithmic IMF slope of newly formed stars (default=1.3)            

       Returns
       -------

       Updates the SFH parameters of the instance
       """  
        assert (
            np.isscalar(start) and np.isscalar(tau) and np.isscalar(met) and np.isscalar(alpha) and np.isscalar(imf_slope) 
            ), \
            "Input parameters must be scalar" 

        # Exponentially declinging SFR
        self.sfr  = np.zeros_like(self.time)
        self.sfr[(self.time <= start)] = np.exp(-(start-self.time[(self.time <= start)])/tau)
        # Normalization to form 1 Msun
        norm = simps(self.sfr,self.time)
        self.sfr  = self.sfr / norm
        # Mass-weights
        dt = (self.time-np.roll(self.time,1))
        dt[0] = dt[1]
        self.wts  = self.sfr * dt
        # Normalized to a total of 1 Msun
        self.wts  = self.wts / np.sum(self.wts)

        # Set as constant the other SFH parameters 
        self.met_evol   = np.zeros_like(self.time) + met
        self.alpha_evol = np.zeros_like(self.time) + alpha
        self.imf_evol = np.zeros_like(self.time) + imf_slope

        # Make sure everything is within the allowed range of models
        assert (self.sfh_check_ssp_range() ),\
          "Input SFH params are outside the MILES model grid you loaded"

    def delayed_tau_sfr(self,start=10, tau=1, met=0, alpha=0, imf_slope=1.3):
        """Delayed exponentially declinging SFR

        This is a standard tau model where the SFR is given by
            SFR(t) = 0 for t < start
            SFR(t) = (start-t) * exp (- (start-t)/tau)) for t >= start
        
        The normalization is such one solar mass is formed

        Parameters
        ----------

        start : scalar (Gyr)
            Start of the burst (default=14)
        tau   : scalar (Gyr)
            e-folding time (default=1)
        met   : scalar (dex)
            (log) Metallicity of newly formed stars (default=0)
        alpha : scalar (dex)
            (log) [alpha/Fe] of newly formed stars (default=0)
        imf_slope : scalar 
            logarithmic IMF slope of newly formed stars (default=1.3)            

       Returns
       -------

       Updates the SFH parameters of the instance
       """  

        assert (
            np.isscalar(start) and np.isscalar(tau) and np.isscalar(met) and np.isscalar(alpha) and np.isscalar(imf_slope) 
            ), \
            "Input parameters must be scalar" 

        # Exponentially declinging SFR
        self.sfr  = np.zeros_like(self.time)
        self.sfr[(self.time <= start)] = (start-self.time[(self.time <= start)]) * np.exp(-(start-self.time[(self.time <= start)])/tau)
        # Normalization to form 1 Msun
        norm = simps(self.sfr,self.time)
        self.sfr  = self.sfr / norm
        # Mass-weights
        dt = (self.time-np.roll(self.time,1))
        dt[0] = dt[1]
        self.wts  = self.sfr * dt
        # Normalized to a total of 1 Msun
        self.wts  = self.wts / np.sum(self.wts)

        # Set as constant the other SFH parameters 
        self.met_evol   = np.zeros_like(self.time) + met
        self.alpha_evol = np.zeros_like(self.time) + alpha
        self.imf_evol = np.zeros_like(self.time) + imf_slope

        # Make sure everything is within the allowed range of models
        assert (self.sfh_check_ssp_range() ),\
          "Input SFH params are outside the MILES model grid you loaded" 

    def lognormal_sfr(self, Tpeak=10, tau=1, met=0, alpha=0, imf_slope=1.3):
        """Lognormal SFR

        The time evolution of the SFR is given by
            SFR(tn) = 1/tn * exp( -(T0-ln(tn))**2 / 2*tau**2)

        Note that tn is in this case the time since the Big Bang and not 
        lookback time as in the SSP. See details in Diemer et al. 2017, 
        ApJ, 839, 26, Appendix A.1
        
        The normalization is such one solar mass is formed

        Parameters
        ----------

        Tpeal : scalar (Gyr)
            Time of the SFR peak (default=14)
        tau   : scalar (Gyr)
            Characteristic time-scale (default=1)
        met   : scalar (dex)
            (log) Metallicity of newly formed stars (default=0)
        alpha : scalar (dex)
            (log) [alpha/Fe] of newly formed stars (default=0)
        imf_slope : scalar 
            logarithmic IMF slope of newly formed stars (default=1.3)            

       Returns
       -------

       Updates the SFH parameters of the instance
       """  

        assert (
            np.isscalar(Tpeak) and np.isscalar(tau) and np.isscalar(met) and np.isscalar(alpha) and np.isscalar(imf_slope) 
            ), \
            "Input parameters must be scalar" 

        # While SSPs are based on lookback times we will use in this case
        # time since the oldest age in the grid (~ time since the Big Bang)
        tn     =  max(self.time)-self.time
        tn[-1] = 1e-4   # Avoid zeros in time...
        Tc     = np.log(max(self.time)-Tpeak)+tau**2
        self.sfr  = np.zeros_like(self.time)
        self.sfr  = (1/tn) * np.exp(-(np.log(tn)-Tc)**2/(2.*tau**2))
        # Normalization to form 1 Msun
        norm = simps(self.sfr,self.time)
        self.sfr  = self.sfr / norm
        # Mass-weights
        dt = (self.time-np.roll(self.time,1))
        dt[0] = dt[1]
        self.wts  = self.sfr * dt
        # Normalized to a total of 1 Msun
        self.wts  = self.wts / np.sum(self.wts)

        # Set as constant the other SFH parameters 
        self.met_evol   = np.zeros_like(self.time) + met
        self.alpha_evol = np.zeros_like(self.time) + alpha
        self.imf_evol = np.zeros_like(self.time) + imf_slope

        # Make sure everything is within the allowed range of models
        assert (self.sfh_check_ssp_range() ),\
          "Input SFH params are outside the MILES model grid you loaded"          

    def double_power_law_sfr(self, a=5, b=5, tp=10, met=0, alpha=0, imf_slope=1.3):
        """Double power law SFR evolution

        The SFR as a function of time is given by (Behroozi et al. 2013)
            SFR(tn) = ((tn/tp)**a + (tn/tp)**b)**-1
        As for the lognormal SFR, tn refers to time since the Big Bang
        
        The normalization is such one solar mass is formed

        Parameters
        ----------

        a     : scalar
            falling slope (default=5)
        b     : scalar
            rising slope (default=5)
        tp   : scalar
            similar to the SFR peak in look-back time (default=10)
        met   : scalar (dex)
            (log) Metallicity of newly formed stars (default=0)
        alpha : scalar (dex)
            (log) [alpha/Fe] of newly formed stars (default=0)
        imf_slope : scalar 
            logarithmic IMF slope of newly formed stars (default=1.3)            

       Returns
       -------

       Updates the SFH parameters of the instance
       """  

        assert (
            np.isscalar(a) and np.isscalar(b) and np.isscalar(tp) and np.isscalar(met) and np.isscalar(alpha) and np.isscalar(imf_slope) 
            ), \
            "Input parameters must be scalar" 

        # While SSPs are based on lookback times we will use in this case
        # time since the oldest age in the grid (~ time since the Big Bang)
        tn     = max(self.time)-self.time
        tau    = max(self.time)-tp
        tn[-1] = 1e-4   # Avoid zeros in time...
        self.sfr  = np.zeros_like(self.time)
        self.sfr  = ((tn/tau)**a + (tn/tau)**-b)**-1
        # Normalization to form 1 Msun
        norm = simps(self.sfr,self.time)
        self.sfr  = self.sfr / norm
        # Mass-weights
        dt = (self.time-np.roll(self.time,1))
        dt[0] = dt[1]
        self.wts  = self.sfr * dt
        # Normalized to a total of 1 Msun
        self.wts  = self.wts / np.sum(self.wts)

        # Set as constant the other SFH parameters 
        self.met_evol   = np.zeros_like(self.time) + met
        self.alpha_evol = np.zeros_like(self.time) + alpha
        self.imf_evol = np.zeros_like(self.time) + imf_slope

        # Make sure everything is within the allowed range of models
        assert (self.sfh_check_ssp_range() ),\
          "Input SFH params are outside the MILES model grid you loaded"          

    def bursty_sfr(self, ages=None, wts=None, mets=None,alphas=None, imfs=None):
        """Bursty star formation history

        The SFH is given by bursts with weights and stellar population 
        parameters defined by the user. 
        
        The normalization is such one solar mass is formed

        Parameters
        ----------

        ages   : array (Gyr)
            Ages of the bursts
        wts    : array
            Weights in mass of the bursts
        mets   : array (dex)
            Metallicities of the bursts (default=0)
        alphas : array (dex)
            [alpha/Fe] of the bursts (default=0)
        imfs : array
            IMF of the bursts (default=1.3)

       Returns
       -------

       Updates the SFH parameters of the instance
       """  
        
        assert(ages), "You forgot to include your bursts" 
        self.time = np.array(ages)

        if not wts:
            self.wts = np.zeros(len(self.time)) + 1
        else:
            self.wts = np.array(wts)
            assert(len(ages) == len(wts)), "Age and weights arrays should be the same length " 
            assert(np.all(self.wts >= 0)),  "Weight array should not be negative" 
        self.wts  = self.wts / np.sum(self.wts)
        # Since the SFR could be ill-define we set it to NaN (it won't be actually used)
        self.sfr = np.zeros(len(self.wts)) + float('NaN')

        if not mets:
            self.met_evol = np.zeros(len(self.time))
        else:   
            assert(len(ages) == len(mets)), "Age and metallicity arrays should be the same length " 
            self.met_evol = np.array(mets)

        if not alphas:
            self.alpha_evol = np.zeros(len(self.time))
        else:   
            assert(len(ages) == len(alphas)), "Age and [alpha/Fe] arrays should be the same length " 
            self.alpha_evol = np.array(alphas)                    

        if not imfs:
            self.imf_evol = np.zeros(len(self.time)) + 1.3
        else:   
            assert(len(ages) == len(imfs)), "Age and IMF arrays should be the same length " 
            self.imf_evol = np.array(imfs)    

        # Make sure everything is within the allowed range of models
        assert (self.sfh_check_ssp_range() ),\
          "Input SFH params are outside the MILES model grid you loaded"

    # This method is effectively the same as bursty_sfh but with a different name. 
    # The idea is that a user may want to have a look at a bursty SFH but the term
    # "user_sfr" may sound too risky. In this case we also assume that the SFH is 
    # constant and therefore the SFR can be calculated as weights/dt
    def user_sfr(self, ages=None, wts=None, mets=None,alphas=None, imfs=None):
        """User-defined star formation history

        The SFH is freely defined by the user.
        
        The normalization is such one solar mass is formed

        Parameters
        ----------

        ages   : array (Gyr)
            Ages of the bursts
        wts    : array 
            Weights in mass of the bursts
        mets   : array (dex)
            Metallicities of the bursts (default=0)
        alphas : array (dex)
            [alpha/Fe] of the bursts (default=0)
        imfs : array
            IMF of the bursts (default=1.3)

       Returns
       -------

       Updates the SFH parameters of the instance
       """  
        
        assert(ages), "You forgot to include your bursts" 
        self.time = np.sort(list(set(ages)))
        dt = (self.time-np.roll(self.time,1))
        dt[0] = dt[1]

        if not wts:
            self.wts = np.zeros(len(self.time)) + 1
        else:
            self.wts = np.array(wts)
            assert(len(ages) == len(wts)), "Age and weight arrays should be the same length " 
            assert(np.all(self.wts >= 0)),  "Weight array should not be negative" 

        self.wts  = self.wts / np.sum(self.wts)
        self.sfr  = self.wts / dt
         # Normalization to form 1 Msun
        norm = simps(self.sfr,self.time)
        self.sfr  = self.sfr / norm

        if not mets:
            self.met_evol = np.zeros(len(self.time))
        else:   
            assert(len(ages) == len(mets)), "Age and metallicity arrays should be the same length " 
            self.met_evol = np.array(mets)

        if not alphas:
            self.alpha_evol = np.zeros(len(self.time))
        else:   
            assert(len(ages) == len(alphas)), "Age and [alpha/Fe] arrays should be the same length " 
            self.alpha_evol = np.array(alphas)                    

        if not imfs:
            self.imf_evol = np.zeros(len(self.time)) + 1.3
        else:   
            assert(len(ages) == len(imfs)), "Age and IMF arrays should be the same length " 
            self.imf_evol = np.array(imfs)    

        # Make sure everything is within the allowed range of models
        assert (self.sfh_check_ssp_range() ),\
          "Input SFH params are outside the MILES model grid you loaded"

    # The methods below are prescriptions for the metallicty, [alpha/Fe], 
    # and IMF time evolution. They do not describe any physical process 
    # but can be useful to describe smooth variations in time. 
    def met_evol_sigmoid(self,start=-1.5,end=0.2,tc=5.,gamma=1.):
        """Sigmoidal metallicity evolution

        The metallicity evolves as a sigmoidal function. This is not
        meant to be physically meaningful but to reproduce the exponential
        character of the chemical evolution
        
        Parameters
        ----------

        start   : scalar (dex)
            Metallicity of the oldest stellar population (default=-1.5)
        end     : scalar (dex)
            Metallicity of the youngest stellar population (default=0.2)
        tc      : scalar (Gyr)
            Characteristic transition time (default=5)
        gamma   : scalar
            Transition slope (default=1)

       Returns
       -------

       Updates the SFH parameters of the instance
       """ 

        assert (
            np.isscalar(start) and np.isscalar(end) and np.isscalar(tc) and np.isscalar(gamma)
            ), \
            "Input parameters must be scalar" 

        assert (
            (max(self.age) >= tc) and (min(self.age) <= tc)
        ), \
            "Transition time is outside of the loaded age range"

        self.met_evol   = (end - start) / (1. +  np.exp(-gamma*(tc-self.time))) + start

        # Make sure everything is within the allowed range of models
        assert (self.sfh_check_ssp_range() ),\
          "Input SFH params are outside the MILES model grid you loaded"

    def met_evol_linear(self,start=-1.5,end=0.2,t_start=5, t_end=1):
        """Linear metallicity evolution

        The metallicity evolves as a ReLU function, i.e., constant at the beginning
        and linearly varing afterwards.
        
        Parameters
        ----------

        start   : scalar (dex)
            Metallicity of the oldest stellar population (default=-1.5)
        end     : scalar (dex)
            Metallicity of the youngest stellar population (default=0.2)
        t_start : scalar (Gyr)
            Start of the metallicity variation (default=5)
        t_end : scalar (Gyr)
            End of the metallicity variation (default=5)
       Returns
       -------

       Updates the SFH parameters of the instance
       """ 

        assert (
            np.isscalar(start) and np.isscalar(end) and np.isscalar(t_start) and np.isscalar(t_end)
            ), \
            "Input parameters must be scalar" 

        assert (
            (max(self.age) >= t_start) and (min(self.age) <= t_start)
        ), \
            "Start time is outside of the loaded age range"

        assert (
            (max(self.age) >= t_end) and (min(self.age) <= t_end)
        ), \
            "End time is outside of the loaded age range"

        slope = (start-end)/(t_start-t_end)

        tgood = (self.time > t_start)
        self.met_evol[tgood] = start 

        tgood = (self.time <= t_start) & (self.time >= t_end)
        self.met_evol[tgood] = slope * (self.time[tgood] - t_start) + start

        tgood =  (self.time < t_end)
        self.met_evol[tgood] = end

        # Make sure everything is within the allowed range of models
        assert (self.sfh_check_ssp_range() ),\
          "Input SFH params are outside the MILES model grid you loaded"

    def alp_evol_sigmoid(self,start=0.4,end=0.0,tc=5.,gamma=1.):
        """Sigmoidal [alpha/Fe] evolution

        The [alpha/Fe] evolves as a sigmoidal function. This is not
        meant to be physically meaningful but to reproduce the exponential
        character of the chemical evolution
        
        Parameters
        ----------

        start   : scalar (dex)
            [alpha/Fe] of the oldest stellar population (default=0.4)
        end     : scalar (dex)
            [alpha/Fe] of the youngest stellar population (default=0.0)
        tc      : scalar (Gyr)
            Characteristic transition time (default=5)
        gamma   : scalar
            Transition slope (default=1)

       Returns
       -------

       Updates the SFH parameters of the instance
       """ 

        assert (
            np.isscalar(start) and np.isscalar(end) and np.isscalar(tc) and np.isscalar(gamma)
            ), \
            "Input parameters must be scalar" 

        assert (
            (max(self.age) >= tc) and (min(self.age) <= tc)
        ), \
            "Transition time is outside of the loaded age range"

        self.alp_evol   = (end - start) / (1. +  np.exp(-gamma*(tc-self.time))) + start

        # Make sure everything is within the allowed range of models
        assert (self.sfh_check_ssp_range() ),\
          "Input SFH params are outside the MILES model grid you loaded"


    def alp_evol_linear(self, start=-1.5,end=0.2,t_start=5, t_end=1):
        """Linear [alpha/Fe]  evolution

        The [alpha/Fe]  evolves as a ReLU function, i.e., constant at the beginning
        and linearly varing afterwards.
        
        Parameters
        ----------

        start   : scalar (dex)
            [alpha/Fe] of the oldest stellar population (default=-1.5)
        end     : scalar (dex)
            [alpha/Fe]  of the youngest stellar population (default=0.2)
        t_start : scalar (Gyr)
            Start of the [alpha/Fe]  variation (default=5)
        t_end : scalar (Gyr)
            End of the [alpha/Fe]  variation (default=1)
       Returns
       -------

       Updates the SFH parameters of the instance
       """ 

        assert (
            np.isscalar(start) and np.isscalar(end) and np.isscalar(t_start) and np.isscalar(t_end)
            ), \
            "Input parameters must be scalar" 

        assert (
            (max(self.age) >= t_start) and (min(self.age) <= t_start)
        ), \
            "Start time is outside of the loaded age range"

        assert (
            (max(self.age) >= t_end) and (min(self.age) <= t_end)
        ), \
            "End time is outside of the loaded age range"

        slope = (start-end)/(t_start-t_end)

        tgood = (self.time > t_start)
        self.alp_evol[tgood] = start 

        tgood = (self.time <= t_start) & (self.time >= t_end)
        self.alp_evol[tgood] = slope * (self.time[tgood] - t_start) + start

        tgood =  (self.time < t_end)
        self.alp_evol[tgood] = end

        # Make sure everything is within the allowed range of models
        assert (self.sfh_check_ssp_range() ),\
          "Input SFH params are outside the MILES model grid you loaded"

    def imf_evol_sigmoid(self,start=0.5,end=3.0,tc=5.,gamma=1.):
        """Sigmoidal IMF slope evolution

        The IMF slope  evolves as a sigmoidal function. This is not
        meant to be physically meaningful but to track the chemical 
        variations (see e.g. Martin-Navarro et al. 2015)
        
        Parameters
        ----------

        start   : scalar (dex)
            IMF slope  of the oldest stellar population (default=0.5)
        end     : scalar (dex)
            IMF slope  of the youngest stellar population (default=3.0)
        tc      : scalar (Gyr)
            Characteristic transition time (default=5)
        gamma   : scalar
            Transition slope (default=1)

       Returns
       -------

       Updates the SFH parameters of the instance
       """ 

        assert (
            np.isscalar(start) and np.isscalar(end) and np.isscalar(tc) and np.isscalar(gamma)
            ), \
            "Input parameters must be scalar" 

        assert (
            (max(self.age) >= tc) and (min(self.age) <= tc)
        ), \
            "Transition time is outside of the loaded age range"

        self.imf_evol   = (end - start) / (1. +  np.exp(-gamma*(tc-self.time))) + start

        # Make sure everything is within the allowed range of models
        assert (self.sfh_check_ssp_range() ),\
          "Input SFH params are outside the MILES model grid you loaded"


    def imf_evol_linear(self, start=-1.5,end=0.2,t_start=5, t_end=1):
        """Linear IMF slope evolution

        The IMF slope evolves as a ReLU function, i.e., constant at the beginning
        and linearly varing afterwards.
        
        Parameters
        ----------

        start   : scalar 
            IMF of the oldest stellar population (default=-1.5)
        end     : scalar 
            IMF  of the youngest stellar population (default=0.2)
        t_start : scalar (Gyr)
            Start of the IMF variation (default=5)
        t_end : scalar (Gyr)
            End of the IMF variation (default=1)
       Returns
       -------

       Updates the SFH parameters of the instance
       """ 

        assert (
            np.isscalar(start) and np.isscalar(end) and np.isscalar(t_start) and np.isscalar(t_end)
            ), \
            "Input parameters must be scalar" 

        assert (
            (max(self.age) >= t_start) and (min(self.age) <= t_start)
        ), \
            "Start time is outside of the loaded age range"

        assert (
            (max(self.age) >= t_end) and (min(self.age) <= t_end)
        ), \
            "End time is outside of the loaded age range"

        slope = (start-end)/(t_start-t_end)

        tgood = (self.time > t_start)
        self.imf_evol[tgood] = start 

        tgood = (self.time <= t_start) & (self.time >= t_end)
        self.imf_evol[tgood] = slope * (self.time[tgood] - t_start) + start

        tgood =  (self.time < t_end)
        self.imf_evol[tgood] = end

        # Make sure everything is within the allowed range of models
        assert (self.sfh_check_ssp_range() ),\
          "Input SFH params are outside the MILES model grid you loaded"
        
    # This method (get_sfh_predictions) is the core of the SFH class. It takes the 
    # normalized input of the SFH instance (namely ages, weights, metallicities, 
    # [alpha/Fe] and IMF and calculates the MILES predictions. Since this class
    # inherits the ssp_models_class, this method also gives photometric prediction 
    # and mean (mass-weighted) stellar population parameters.
    def get_sfh_predictions(self):
        """Derives the MILES predictions for the desired SFH

        The spectrum associated to the SFH is calculated using the 
        (normalized) mass weights. This spectrum is then used to 
        calculate the photometric predictions that are also given.
        Mass-weighted stellar population parameter are also calculated
        
        Parameters
        ----------
        self : SFH instance call

        Returns
        -------

        MILES predictions
        """ 
        
        # We make a copy of the instance to speed up the interpolation
        intp = copy(self)
        out  = copy(self)

        # We make an initial call to get_ssp_params (ssp_model_class) to 
        # obtain the values of the triangulation to be used extensively below
        tmp  = self.get_ssp_by_params(age=self.time[0], met=self.met_evol[0], imf_slope=self.imf_evol[0], alpha=self.alpha_evol[0])
        nspec_in      = intp.spec.shape[1]

        # Making sure we select the right IMF and alpha models
        uimf_slope = np.unique(self.imf_slope)
        nimf_slope = len(uimf_slope)
        ospec = np.zeros(len(np.squeeze(tmp.spec)))
        
        # We iterate now over all the age bins in the SFH 
        for t, date in enumerate(self.time): 

            ok = (nimf_slope == 1)
            if self.alpha_flag == 1:
                if ok == True:
                    input_pt = np.array([self.time[t],self.met_evol[t]], ndmin=2)
                else:
                    input_pt = np.array([self.time[t],self.met_evol[t],self.imf_evol[t]], ndmin=2)
            else:
                if ok == True:
                    input_pt = np.array([self.time[t],self.met_evol[t],self.alpha_evol[t]], ndmin=2)
                else:
                    input_pt = np.array([self.time[t],self.met_evol[t],self.imf_evol[t],self.alpha_evol[t]], ndmin=2)

            vtx, wts = utils.interp_weights(tmp.params, input_pt, tmp.tri)
            vtx, wts = vtx.ravel(), wts.ravel()

            # Update quantities
            keys = list(out.main_keys)
            for i in range(len(out.main_keys)):
                if (keys[i] == 'wave') or (keys[i] == 'spec') or (keys[i] == 'nspec') or \
                    (keys[i] == 'age')  or (keys[i] == 'met') or (keys[i] == 'alpha') or  \
                    (keys[i] == 'imf_slope') or (keys[i] == 'filename') or (keys[i] == 'isochrone') or \
                    (keys[i] == 'imf_type') or (keys[i] == 'index'):
                    continue
                
                # pre_val is the quantity to be updated. intp_val contains all the models and it
                # is used for the interpolation
                pre_val  = np.array(getattr(out,keys[i]))
                intp_val = np.array(getattr(intp,keys[i]))
                
                if (np.ndim(intp_val) == 0):
                    continue
                
                # This is where the interpolation happens using the triangulation calculated above and
                # over the models stored in intp_val. Note that the final value of each key 
                # is the mass weighted combination 
                # keys_final = keys_t=0 * weights_t=0 + .... + keys_t=n * weights_t=n
                # where n is the final (oldest) age in the SFH
                if (intp_val.shape[0] == nspec_in):
                    if t == 0:
                        setattr(out, keys[i], np.dot(intp_val[vtx],wts)* self.wts[t])
                    else:
                        setattr(out, keys[i], pre_val + np.dot(intp_val[vtx],wts)* self.wts[t])

            # The final spectrum is also the mass-weighted one
            ospec = ospec + np.dot(intp.spec[:,vtx],wts) * self.wts[t]       

        # Formating the instance so it can be used latter
        out.spec      = np.array(ospec,ndmin=2).T
        out.age       = np.sum(self.time*self.wts)
        out.met       = np.sum(self.met_evol*self.wts)
        out.alpha     = np.sum(self.alpha_evol*self.wts) 
        out.imf_slope = np.sum(self.imf_evol*self.wts)
        out.nspec     = 1
        out.filename  = None
        out.isochrone = out.isochrone[0]
        out.imf_type  = out.imf_type[0]
        out.index     = np.nan     

        return out

    def sfh_check_ssp_range(self):
        """Checks if expected SSPs are within the model grid

        When calculating the parameters of the SFH this method
        makes sure all the SSPs are within the array of available
        MILES models.

        Parameters
        ----------
        self : SFH instance call

        Returns
        ---------- 
        safe : boolean
            True if all parameters are within the allowed range,
            False otherwise

        """
        uimf_slope = np.unique(self.imf_slope)
        nimf_slope = len(uimf_slope)
        ok = (nimf_slope == 1)

        if self.alpha_flag == 1:
            if ok == 1:
                if (
                    ( max(self.time) <=  max(self.age) ) and
                    ( min(self.time) >=  min(self.age) ) and
                    ( max(self.met_evol) <=  max(self.met) ) and
                    ( min(self.met_evol) >=  min(self.met) ) 
                 ): safe = True
                else:
                    safe = False
            else:
                if (
                    ( max(self.time) <=  max(self.age) ) and
                    ( min(self.time) >=  min(self.age) ) and
                    ( max(self.met_evol) <=  max(self.met) ) and
                    ( min(self.met_evol) >=  min(self.met) ) and
                    ( max(self.imf_evol) <=  max(self.imf_slope) ) and
                    ( min(self.imf_evol) >=  min(self.imf_slope) ) 
                 ): safe = True
                else:
                    safe = False
        else:
            if ok == 1:
                if (
                    ( max(self.time) <=  max(self.age) ) and
                    ( min(self.time) >=  min(self.age) ) and
                    ( max(self.alpha_evol) <=  max(self.alpha) ) and
                    ( min(self.alpha_evol) >=  min(self.alpha) ) and
                    ( max(self.met_evol) <=  max(self.met) ) and
                    ( min(self.met_evol) >=  min(self.met) ) 
                 ): safe = True
                else:
                    safe = False
            else:
                if (
                    ( max(self.time) <=  max(self.age) ) and
                    ( min(self.time) >=  min(self.age) ) and
                    ( max(self.met_evol) <=  max(self.met) ) and
                    ( min(self.met_evol) >=  min(self.met) ) and
                    ( max(self.alpha_evol) <=  max(self.alpha) ) and
                    ( min(self.alpha_evol) >=  min(self.alpha) ) and
                    ( max(self.imf_evol) <=  max(self.imf_slope) ) and
                    ( min(self.imf_evol) >=  min(self.imf_slope) ) 
                 ): safe = True
                else:
                    safe = False

        return safe

    def sfh_normalize(self):
        """Normalizes weights and SFR to one solar mass

        This method is not used internally but it can be used by the 
        user to combine different SFHs into a single prediction

        Parameters
        ----------
        self : SFH instance call

        Returns
        ---------- 

        Updates the SFH parameters of the instance
        """

         # Normalization to form 1 Msun
        self.wts  = self.wts / np.sum(self.wts)
        norm = simps(self.sfr,self.time)
        self.sfr  = self.sfr / norm