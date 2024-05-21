# -*- coding: utf-8 -*-
import logging
from copy import copy

import numpy as np
from scipy.integrate import simps

import pymiles.pymiles_utils as utils
from pymiles.ssp_models import ssp_models

# from ipdb import set_trace as stop
# ==============================================================================

logger = logging.getLogger("pymiles.sfh")


class sfh(ssp_models):
    # -----------------------------------------------------------------------------
    def __init__(
        self,
        source="MILES_SSP",
        version="9.1",
        isochrone="T",
        imf_type="ch",
        alp_type="fix",
        show_tree=False,
        verbose=False,
    ):
        """
        Creates an instance of the class

        Parameters
        ----------
        source:    Name of input models to use.
                   Valid inputs are MILES_SSP/CaT_SSP/EMILES_SSP
        version:   version number of the models
        isochrone: Type of isochrone to use. Valid inputs are P/T for Padova+00
                   and BaSTI isochrones respectively (Default: T)
        imf_type:  Type of IMF shape. Valid inputs are ch/ku/kb/un/bi (Default: ch)
        alp_type:  Type of [alpha/Fe]. Valid inputs are fix/variable (Default: fix).
                   Variable [alpha/Fe] predictions are only available for
                   BaSTI isochrones.
        show_tree: Bool that shows the variables available with the instance
        verbose:   Flag to verbose mode

        Notes
        -----
        We limit the choice of models to a given isochrone and imt_type for
        effective loading.
        Otherwise it can take along time to upload the entire dataset

        Returns
        -------
        None
            Object instance

        """
        # Inheriting classes
        ssp_models.__init__(
            self,
            source=source,
            version=version,
            isochrone=isochrone,
            imf_type=imf_type,
            alp_type=alp_type,
        )

        # The SFH is defined based on a time array
        self.time = np.sort(list(set(self.age)))
        self.wts = np.zeros_like(self.time)
        self.sfr = np.zeros_like(self.time)
        self.alpha_evol = np.zeros_like(self.time)
        self.imf_evol = np.zeros_like(self.time) + 1.3
        self.met_evol = np.zeros_like(self.time)

        # Flagging if all elements of alpha are NaNs
        if alp_type == "fix":
            self.fixed_alpha = True
        elif alp_type == "variable":
            self.fixed_alpha = False
        else:
            raise ValueError("alp_type should be 'fix' or 'variable'")

        self.main_keys = list(self.__dict__.keys())

        logger.info(source + " models loaded")

    @staticmethod
    def _process_param(argname, arg, refname, ref, offset=0):
        if np.isscalar(arg):
            arg = np.full_like(ref, arg)
        elif len(arg) == len(ref) + offset:
            arg = np.array(arg)
        else:
            raise ValueError(
                f"{refname} and {argname} arrays should be the same length"
            )

    @staticmethod
    def _validate_scalar(arg, argname="Input"):
        if not np.isscalar(arg):
            raise ValueError(f"{argname} should be scalar")

    @staticmethod
    def _validate_in_range(arg, low, high, argname="Input"):
        if arg < low or arg > high:
            raise ValueError(f"{argname} is out of range")

    def _validate_sfh(self):
        if not self.sfh_check_ssp_range():
            raise ValueError(
                "Input SFH params are outside the MILES model grid you loaded"
            )

    # ----------
    # Methods to define the time array. This is the fundation of
    # the SFHs as it defines where the SSPs will be evalated
    # ----------
    def set_time_range(self, start=14.0, end=0.0):
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
        None
            Sets the age array over which the SFH will be constructed and
            update the rest of the variables accordingly
        """

        self._validate_scalar(start, "Start")
        self._validate_scalar(end, "End")
        if start < end:
            raise ValueError("Start date must be older than end date")

        idx = (self.age <= start) & (self.age >= end)
        self.time = np.sort(list(set(self.age[idx])))
        self.met_evol = np.zeros_like(self.time)
        self.alpha_evol = np.zeros_like(self.time)
        self.imf_evol = np.zeros_like(self.time) + 1.3

    def set_time_user(self, user_time):
        """User-defined age array

        Sets the input array as the time axis for the SFH

        Parameters
        ----------

        user_time : array (Gyr)
             User-defined array of ages to evaluate the SFH

        Returns
        -------
        None
            Sets the age array over which the SFH will be constructed and
            update the rest of the variables accordingly
        """
        assert len(user_time) >= 2, (
            "You should not be here... only one age was provided."
            "Look for SSP models instead."
        )

        self.time = np.sort(list(set(user_time)))
        self.met_evol = np.zeros_like(self.time)
        self.alpha_evol = np.zeros_like(self.time)
        self.imf_evol = np.zeros_like(self.time) + 1.3

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
    def constant_sfr(self, start=14, end=1.0, met=0.0, alpha=0, imf_slope=1.3):
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
        None
            Updates the SFH parameters of the instance
        """

        for inp in (start, end, met, alpha, imf_slope):
            self._validate_scalar(inp)

        if start < end:
            raise ValueError("Start time must happen befor the end of the burst")

        # Constant SFR
        self.sfr = np.zeros_like(self.time)
        self.sfr[(self.time <= start) & (self.time >= end)] = 1
        # Normalization to form 1 Msun
        norm = simps(self.sfr, self.time)
        self.sfr = self.sfr / norm
        # Mass-weights
        dt = self.time - np.roll(self.time, 1)
        dt[0] = dt[1]
        self.wts = self.sfr * dt
        # Normalized to a total of 1 Msun
        self.wts = self.wts / np.sum(self.wts)

        # Set as constant the other SFH parameters
        self.met_evol = np.zeros_like(self.time) + met
        self.alpha_evol = np.zeros_like(self.time) + alpha
        self.imf_evol = np.zeros_like(self.time) + imf_slope

        # Make sure everything is within the allowed range of models
        self._validate_sfh()

    def tau_sfr(self, start=10, tau=1, met=0, alpha=0, imf_slope=1.3):
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
        None
            Updates the SFH parameters of the instance
        """
        for inp in (start, tau, met, alpha, imf_slope):
            self._validate_scalar(inp)

        # Exponentially declinging SFR
        self.sfr = np.zeros_like(self.time)
        self.sfr[(self.time <= start)] = np.exp(
            -(start - self.time[(self.time <= start)]) / tau
        )
        # Normalization to form 1 Msun
        norm = simps(self.sfr, self.time)
        self.sfr = self.sfr / norm
        # Mass-weights
        dt = self.time - np.roll(self.time, 1)
        dt[0] = dt[1]
        self.wts = self.sfr * dt
        # Normalized to a total of 1 Msun
        self.wts = self.wts / np.sum(self.wts)

        # Set as constant the other SFH parameters
        self.met_evol = np.zeros_like(self.time) + met
        self.alpha_evol = np.zeros_like(self.time) + alpha
        self.imf_evol = np.zeros_like(self.time) + imf_slope

        # Make sure everything is within the allowed range of models
        self._validate_sfh()

    def delayed_tau_sfr(self, start=10, tau=1, met=0, alpha=0, imf_slope=1.3):
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
        None
            Updates the SFH parameters of the instance
        """

        for inp in (start, tau, met, alpha, imf_slope):
            self._validate_scalar(inp)

        # Exponentially declinging SFR
        self.sfr = np.zeros_like(self.time)
        self.sfr[(self.time <= start)] = (
            start - self.time[(self.time <= start)]
        ) * np.exp(-(start - self.time[(self.time <= start)]) / tau)
        # Normalization to form 1 Msun
        norm = simps(self.sfr, self.time)
        self.sfr = self.sfr / norm
        # Mass-weights
        dt = self.time - np.roll(self.time, 1)
        dt[0] = dt[1]
        self.wts = self.sfr * dt
        # Normalized to a total of 1 Msun
        self.wts = self.wts / np.sum(self.wts)

        # Set as constant the other SFH parameters
        self.met_evol = np.zeros_like(self.time) + met
        self.alpha_evol = np.zeros_like(self.time) + alpha
        self.imf_evol = np.zeros_like(self.time) + imf_slope

        # Make sure everything is within the allowed range of models
        self._validate_sfh()

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
        None
            Updates the SFH parameters of the instance
        """

        for inp in (Tpeak, tau, met, alpha, imf_slope):
            self._validate_scalar(inp)

        # While SSPs are based on lookback times we will use in this case
        # time since the oldest age in the grid (~ time since the Big Bang)
        tn = max(self.time) - self.time
        tn[-1] = 1e-4  # Avoid zeros in time...
        Tc = np.log(max(self.time) - Tpeak) + tau**2
        self.sfr = np.zeros_like(self.time)
        self.sfr = (1 / tn) * np.exp(-((np.log(tn) - Tc) ** 2) / (2.0 * tau**2))
        # Normalization to form 1 Msun
        norm = simps(self.sfr, self.time)
        self.sfr = self.sfr / norm
        # Mass-weights
        dt = self.time - np.roll(self.time, 1)
        dt[0] = dt[1]
        self.wts = self.sfr * dt
        # Normalized to a total of 1 Msun
        self.wts = self.wts / np.sum(self.wts)

        # Set as constant the other SFH parameters
        self.met_evol = np.zeros_like(self.time) + met
        self.alpha_evol = np.zeros_like(self.time) + alpha
        self.imf_evol = np.zeros_like(self.time) + imf_slope

        # Make sure everything is within the allowed range of models
        self._validate_sfh()

    def double_power_law_sfr(self, a=5, b=5, tp=10, met=0, alpha=0, imf_slope=1.3):
        """Double power law SFR evolution

        The SFR as a function of time is given by (Behroozi et al. 2013)
        .. math:: SFR(tn) = ((tn/tp)**a + (tn/tp)**b)**-1

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
        None
            Updates the SFH parameters of the instance
        """

        for inp in (a, b, tp, met, alpha, imf_slope):
            self._validate_scalar(inp)

        # While SSPs are based on lookback times we will use in this case
        # time since the oldest age in the grid (~ time since the Big Bang)
        tn = max(self.time) - self.time
        tau = max(self.time) - tp
        tn[-1] = 1e-4  # Avoid zeros in time...
        self.sfr = np.zeros_like(self.time)
        self.sfr = ((tn / tau) ** a + (tn / tau) ** -b) ** -1
        # Normalization to form 1 Msun
        norm = simps(self.sfr, self.time)
        self.sfr = self.sfr / norm
        # Mass-weights
        dt = self.time - np.roll(self.time, 1)
        dt[0] = dt[1]
        self.wts = self.sfr * dt
        # Normalized to a total of 1 Msun
        self.wts = self.wts / np.sum(self.wts)

        # Set as constant the other SFH parameters
        self.met_evol = np.zeros_like(self.time) + met
        self.alpha_evol = np.zeros_like(self.time) + alpha
        self.imf_evol = np.zeros_like(self.time) + imf_slope

        # Make sure everything is within the allowed range of models
        self._validate_sfh()

    def bursty_sfr(self, ages=None, wts=None, mets=0.0, alphas=0.0, imfs=1.3):
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
        None
            Updates the SFH parameters of the instance
        """

        if not ages:
            raise ValueError("You forgot to include your bursts")
        self.time = np.array(ages)

        self.wts = wts
        self._process_param("Weights", self.wts, "Age", self.age, offset=1)
        if np.any(self.wts < 0):
            raise ValueError("Weight array should not be negative")
        self.wts = self.wts / np.sum(self.wts)

        self.met_evol = mets
        self._process_param("Metallicity", self.met_evol, "Age", self.age)

        self.alpha_evol = alphas
        self._process_param("[alpha/Fe]", self.alpha_evol, "Age", self.age)

        self.imf_evol = imfs
        self._process_param("IMF", self.imf_evol, "Age", self.age)

        # Since the SFR could be ill-define we set it to NaN (it won't be actually used)
        self.sfr = np.zeros(len(self.wts)) + float("NaN")

        # Make sure everything is within the allowed range of models
        self._validate_sfh()

    # This method is effectively the same as bursty_sfh but with a different name.
    # The idea is that a user may want to have a look at a bursty SFH but the term
    # "user_sfr" may sound too risky. In this case we also assume that the SFH is
    # constant and therefore the SFR can be calculated as weights/dt

    def user_sfr(self, ages=None, wts=None, mets=None, alphas=None, imfs=None):
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
        None
            Updates the SFH parameters of the instance
        """
        if not ages:
            raise ValueError("You forgot to include your bursts")
        self.time = np.array(ages)

        self.wts = wts
        self._process_param("Weights", self.wts, "Age", self.age, offset=1)
        if np.any(self.wts < 0):
            raise ValueError("Weight array should not be negative")
        self.wts = self.wts / np.sum(self.wts)

        self.met_evol = mets
        self._process_param("Metallicity", self.met_evol, "Age", self.age)

        self.alpha_evol = alphas
        self._process_param("[alpha/Fe]", self.alpha_evol, "Age", self.age)

        self.imf_evol = imfs
        self._process_param("IMF", self.imf_evol, "Age", self.age)

        self.time = np.sort(list(set(ages)))
        dt = self.time - np.roll(self.time, 1)
        dt[0] = dt[1]

        self.wts = self.wts / np.sum(self.wts)
        self.sfr = self.wts / dt
        # Normalization to form 1 Msun
        norm = simps(self.sfr, self.time)
        self.sfr = self.sfr / norm

        # Make sure everything is within the allowed range of models
        self._validate_sfh()

    # The methods below are prescriptions for the metallicty, [alpha/Fe],
    # and IMF time evolution. They do not describe any physical process
    # but can be useful to describe smooth variations in time.
    def met_evol_sigmoid(self, start=-1.5, end=0.2, tc=5.0, gamma=1.0):
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
        None
            Updates the SFH parameters of the instance
        """

        for inp in (start, end, tc, gamma):
            self._validate_scalar(inp)

        self._validate_in_range(tc, min(self.age), max(self.age), "Transition time")

        self.met_evol = (end - start) / (
            1.0 + np.exp(-gamma * (tc - self.time))
        ) + start

        # Make sure everything is within the allowed range of models
        self._validate_sfh()

    def met_evol_linear(self, start=-1.5, end=0.2, t_start=5, t_end=1):
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
        None
            Updates the SFH parameters of the instance
        """

        for inp in (start, end, t_start, t_end):
            self._validate_scalar(inp)

        self._validate_in_range(t_start, min(self.age), max(self.age), "Start time")
        self._validate_in_range(t_end, min(self.age), max(self.age), "End time")

        slope = (start - end) / (t_start - t_end)

        tgood = self.time > t_start
        self.met_evol[tgood] = start

        tgood = (self.time <= t_start) & (self.time >= t_end)
        self.met_evol[tgood] = slope * (self.time[tgood] - t_start) + start

        tgood = self.time < t_end
        self.met_evol[tgood] = end

        # Make sure everything is within the allowed range of models
        self._validate_sfh()

    def alp_evol_sigmoid(self, start=0.4, end=0.0, tc=5.0, gamma=1.0):
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
        None
            Updates the SFH parameters of the instance
        """

        for inp in (start, end, tc, gamma):
            self._validate_scalar(inp)

        self._validate_in_range(tc, min(self.age), max(self.age), "Transition time")

        self.alp_evol = (end - start) / (
            1.0 + np.exp(-gamma * (tc - self.time))
        ) + start

        # Make sure everything is within the allowed range of models
        self._validate_sfh()

    def alp_evol_linear(self, start=-1.5, end=0.2, t_start=5, t_end=1):
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
        None
            Updates the SFH parameters of the instance
        """

        for inp in (start, end, t_start, t_end):
            self._validate_scalar(inp)

        self._validate_in_range(t_start, min(self.age), max(self.age), "Start time")
        self._validate_in_range(t_end, min(self.age), max(self.age), "End time")

        slope = (start - end) / (t_start - t_end)

        tgood = self.time > t_start
        self.alp_evol[tgood] = start

        tgood = (self.time <= t_start) & (self.time >= t_end)
        self.alp_evol[tgood] = slope * (self.time[tgood] - t_start) + start

        tgood = self.time < t_end
        self.alp_evol[tgood] = end

        # Make sure everything is within the allowed range of models
        self._validate_sfh()

    def imf_evol_sigmoid(self, start=0.5, end=3.0, tc=5.0, gamma=1.0):
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
        None
            Updates the SFH parameters of the instance
        """

        for inp in (start, end, tc, gamma):
            self._validate_scalar(inp)

        self._validate_in_range(tc, min(self.age), max(self.age), "Transition time")

        self.imf_evol = (end - start) / (
            1.0 + np.exp(-gamma * (tc - self.time))
        ) + start

        # Make sure everything is within the allowed range of models
        self._validate_sfh()

    def imf_evol_linear(self, start=-1.5, end=0.2, t_start=5, t_end=1):
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
        None
            Updates the SFH parameters of the instance
        """

        for inp in (start, end, t_start, t_end):
            self._validate_scalar(inp)

        self._validate_in_range(t_start, min(self.age), max(self.age), "Start time")
        self._validate_in_range(t_end, min(self.age), max(self.age), "End time")

        slope = (start - end) / (t_start - t_end)

        tgood = self.time > t_start
        self.imf_evol[tgood] = start

        tgood = (self.time <= t_start) & (self.time >= t_end)
        self.imf_evol[tgood] = slope * (self.time[tgood] - t_start) + start

        tgood = self.time < t_end
        self.imf_evol[tgood] = end

        # Make sure everything is within the allowed range of models
        self._validate_sfh()

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
        ssp_models
            MILES predictions
        """

        # We make a copy of the instance to speed up the interpolation
        intp = copy(self)
        out = copy(self)

        # We make an initial call to get_ssp_params (ssp_model_class) to
        # obtain the values of the triangulation to be used extensively below
        tmp = self.get_ssp_by_params(
            age=self.time[0],
            met=self.met_evol[0],
            imf_slope=self.imf_evol[0],
            alpha=self.alpha_evol[0],
        )
        nspec_in = intp.spec.shape[1]

        # Making sure we select the right IMF and alpha models
        uimf_slope = np.unique(self.imf_slope)
        nimf_slope = len(uimf_slope)
        ospec = np.zeros(len(np.squeeze(tmp.spec)))

        # We iterate now over all the age bins in the SFH
        for t, date in enumerate(self.time):
            ok = nimf_slope == 1
            if self.fixed_alpha:
                if ok:
                    input_pt = np.array([self.time[t], self.met_evol[t]], ndmin=2)
                else:
                    input_pt = np.array(
                        [self.time[t], self.met_evol[t], self.imf_evol[t]], ndmin=2
                    )
            else:
                if ok:
                    input_pt = np.array(
                        [self.time[t], self.met_evol[t], self.alpha_evol[t]], ndmin=2
                    )
                else:
                    input_pt = np.array(
                        [
                            self.time[t],
                            self.met_evol[t],
                            self.imf_evol[t],
                            self.alpha_evol[t],
                        ],
                        ndmin=2,
                    )

            vtx, wts = utils.interp_weights(tmp.params, input_pt, tmp.tri)
            vtx, wts = vtx.ravel(), wts.ravel()

            # Update quantities
            keys = list(out.main_keys)
            for i in range(len(out.main_keys)):
                if (
                    (keys[i] == "wave")
                    or (keys[i] == "spec")
                    or (keys[i] == "nspec")
                    or (keys[i] == "age")
                    or (keys[i] == "met")
                    or (keys[i] == "alpha")
                    or (keys[i] == "imf_slope")
                    or (keys[i] == "filename")
                    or (keys[i] == "isochrone")
                    or (keys[i] == "imf_type")
                    or (keys[i] == "index")
                ):
                    continue

                # pre_val is the quantity to be updated. intp_val contains all
                # the models and it is used for the interpolation
                pre_val = np.array(getattr(out, keys[i]))
                intp_val = np.array(getattr(intp, keys[i]))

                if np.ndim(intp_val) == 0:
                    continue

                # This is where the interpolation happens using the
                # triangulation calculated above and over the models stored in
                # intp_val. Note that the final value of each key is the mass
                # weighted combination
                # keys_final = keys_t=0 * weights_t=0 + .... + keys_t=n * weights_t=n
                # where n is the final (oldest) age in the SFH
                if intp_val.shape[0] == nspec_in:
                    if t == 0:
                        setattr(out, keys[i], np.dot(intp_val[vtx], wts) * self.wts[t])
                    else:
                        setattr(
                            out,
                            keys[i],
                            pre_val + np.dot(intp_val[vtx], wts) * self.wts[t],
                        )

            # The final spectrum is also the mass-weighted one
            ospec = ospec + np.dot(intp.spec[:, vtx], wts) * self.wts[t]

        # Formating the instance so it can be used latter
        out.spec = np.array(ospec, ndmin=2).T
        out.age = np.sum(self.time * self.wts)
        out.met = np.sum(self.met_evol * self.wts)
        out.alpha = np.sum(self.alpha_evol * self.wts)
        out.imf_slope = np.sum(self.imf_evol * self.wts)
        out.nspec = 1
        out.filename = None
        out.isochrone = out.isochrone[0]
        out.imf_type = out.imf_type[0]
        out.index = np.nan

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
        -------
        safe : boolean
            True if all parameters are within the allowed range,
            False otherwise

        """
        if (
            (max(self.time) <= max(self.age))
            and (min(self.time) >= min(self.age))
            and (max(self.met_evol) <= max(self.met))
            and (min(self.met_evol) >= min(self.met))
            and (max(self.alpha_evol) <= max(self.alpha))
            and (min(self.alpha_evol) >= min(self.alpha))
            and (max(self.imf_evol) <= max(self.imf_slope))
            and (min(self.imf_evol) >= min(self.imf_slope))
        ):
            safe = True
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
        -------
        None
            Updates the SFH parameters of the instance
        """

        # Normalization to form 1 Msun
        self.wts = self.wts / np.sum(self.wts)
        norm = simps(self.sfr, self.time)
        self.sfr = self.sfr / norm
