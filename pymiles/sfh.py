# -*- coding: utf-8 -*-
import logging

import numpy as np
from astropy import units as u
from scipy.integrate import simpson

logger = logging.getLogger("pymiles.sfh")


class SFH:
    """
    Class for manipulating star formation histories (SFH) and create derived spectra

    Attributes
    ----------
    time: astropy.units.Quantity
        Look-back time
    sfr: astropy.units.Quantity
        Values of the star formation rate (SFR) at each time
    met: np.ndarray
        Values of the metallicity at each time
    alpha: np.ndarray
        Values of alpha/Fe at each time
    imf: np.ndarray
        Values of the IMF slope
    """

    time: u.Quantity = np.linspace(0.035, 13.5, 20) * u.Gyr
    sfr: u.Quantity = np.zeros(20) * u.Msun / u.Gyr
    met: np.ndarray = np.zeros(20)
    alpha: np.ndarray = np.zeros(20)
    imf: np.ndarray = np.full(20, 1.3)

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
        if np.ndim(arg) != 0:
            raise ValueError(f"{argname} should be scalar")

    @staticmethod
    def _validate_in_range(arg, low, high, argname="Input"):
        if arg < low or arg > high:
            raise ValueError(f"{argname} is out of range")

    def _normalize(self, sfr):
        norm = simpson(sfr, x=self.time.to_value(u.yr))
        self.sfr = sfr / norm * self.mass / u.yr

        # Mass-weights
        dt = self.time - np.roll(self.time, 1)
        dt[0] = dt[1]
        self.time_weights = (self.sfr * dt).to(u.Msun)

    def tau_sfr(self, start=10.0 * u.Gyr, tau=1.0 * u.Gyr, mass=1.0 * u.Msun):
        """Exponentially declinging SFR

        This is a standard tau model where the SFR is given by
            SFR(t) = 0 for t < start
            SFR(t) = exp (- (t-start)/tau)) for t >= start

        Parameters
        ----------

        start : astroy.units.Quantity
            Start of the burst (default=14)
        tau   : astroy.units.Quantity
            e-folding time (default=1)
        mass : astropy.units.Quantity

        Returns
        -------
        None
            Updates the SFH parameters of the instance
        """
        for inp in (start, tau, mass):
            self._validate_scalar(inp)

        # Exponentially declinging SFR
        self.time <= start
        sfr = np.zeros(self.time.shape)
        sfr[(self.time <= start)] = np.exp(
            -(start - self.time[(self.time <= start)]) / tau
        )
        self.mass = mass
        self._normalize(sfr)

    def delayed_tau_sfr(self, start=10.0 * u.Gyr, tau=1.0 * u.Gyr, mass=1.0 * u.Msun):
        """Delayed exponentially declinging SFR

        This is a standard tau model where the SFR is given by
            SFR(t) = 0 for t < start
            SFR(t) = (start-t) * exp (- (start-t)/tau)) for t >= start


        Parameters
        ----------

        start : scalar (Gyr)
            Start of the burst (default=14)
        tau   : scalar (Gyr)
            e-folding time (default=1)

        Returns
        -------
        None
            Updates the SFH parameters of the instance
        """

        for inp in (start, tau):
            self._validate_scalar(inp)

        sfr = np.zeros(self.time.shape)
        sfr[(self.time <= start)] = (start - self.time[(self.time <= start)]) * np.exp(
            -(start - self.time[(self.time <= start)]) / tau
        )

        self.mass = mass
        self._normalize(sfr)

    def lognormal_sfr(self, Tpeak=10.0 * u.Gyr, tau=1.0 * u.Gyr, mass=1.0 * u.Msun):
        """Lognormal SFR

        The time evolution of the SFR is given by
            SFR(tn) = 1/tn * exp( -(T0-ln(tn))**2 / 2*tau**2)

        Note that tn is in this case the time since the Big Bang and not
        lookback time as in the SSP. See details in Diemer et al. 2017,
        ApJ, 839, 26, Appendix A.1


        Parameters
        ----------

        Tpeak : scalar (Gyr)
            Time of the SFR peak (default=14)
        tau   : scalar (Gyr)
            Characteristic time-scale (default=1)

        Returns
        -------
        None
            Updates the SFH parameters of the instance
        """

        for inp in (Tpeak, tau):
            self._validate_scalar(inp)

        time = self.time.to_value(u.Gyr)
        tau = tau.to_value(u.Gyr)
        Tpeak = Tpeak.to_value(u.Gyr)
        self.mass = mass

        # While SSPs are based on lookback times we will use in this case
        # time since the oldest age in the grid (~ time since the Big Bang)
        tn = np.max(time) - time
        tn[-1] = 1e-4  # Avoid zeros in time...
        Tc = np.log(time.max() - Tpeak) + tau**2
        sfr = np.zeros(self.time.shape)
        sfr = (1 / tn) * np.exp(-((np.log(tn) - Tc) ** 2) / (2.0 * tau**2))
        # Normalization to form 1 Msun
        self._normalize(sfr)

    def double_power_law_sfr(self, a=5, b=5, tp=10 * u.Gyr, mass=1.0 * u.Msun):
        """Double power law SFR evolution

        The SFR as a function of time is given by (Behroozi et al. 2013)
        .. math:: SFR(tn) = ((tn/tp)**a + (tn/tp)**-b)**-1

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
        Returns
        -------
        None
            Updates the SFH parameters of the instance
        """

        for inp in (a, b, tp):
            self._validate_scalar(inp)

        self.mass = mass

        # While SSPs are based on lookback times we will use in this case
        # time since the oldest age in the grid (~ time since the Big Bang)
        tn = max(self.time) - self.time
        tau = max(self.time) - tp
        tn[-1] = 1e-4 * u.Gyr  # Avoid zeros in time...
        sfr = np.zeros_like(self.time)
        sfr = ((tn / tau) ** a + (tn / tau) ** -b) ** -1

        self._normalize(sfr)

    def set_sfr(self, sfr, mass=None):
        """User-defined SFR

        Parameters
        ----------
        sfr     : np.ndarray
            Star formation rate array, with the same shape as `time`.
        mass     : scalar
            If given, normalize the input SFR such that it produces this mass.

        Returns
        -------
        None
            Updates the SFH parameters of the instance
        """
        if mass is None:
            self._set_input_sfr(sfr)
        else:
            self.mass = mass
            self._normalize(sfr)

    @u.quantity_input
    def _set_input_sfr(self, sfr: u.Quantity[u.Msun / u.Gyr]):
        # For some reason simpson does not maintain the units information so we
        # need to do the conversion
        self.mass = (
            simpson(sfr.to_value(u.Msun / u.yr), x=self.time.to_value(u.yr)) * u.Msun
        )
        print(self.mass)
        dt = self.time - np.roll(self.time, 1)
        dt[0] = dt[1]
        self.time_weights = (sfr * dt).to(u.Msun)
        self.sfr = sfr

    @staticmethod
    def _linear(time, start, end, t_start, t_end):
        for inp in (start, end, t_start, t_end):
            SFH._validate_scalar(inp)

        slope = (start - end) / (t_start - t_end)
        out = np.empty(time.shape)

        m = time > t_start
        out[m] = start

        m = (time <= t_start) & (time >= t_end)
        out[m] = slope * (time[m] - t_start) + start

        m = time < t_end
        out[m] = end

        return out

    @staticmethod
    def _sigmoid(time, start, end, tc, gamma):
        for inp in (start, end, tc, gamma):
            SFH._validate_scalar(inp)

        return (end - start) / (1.0 + np.exp(-gamma * (tc - time))) + start

    def sigmoid_met(self, start=-1.5, end=0.2, tc=5.0 * u.Gyr, gamma=1.0 / u.Gyr):
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
        self.met = self._sigmoid(start, end, tc, gamma)

    def sigmoid_alpha(self, start=0.4, end=0.0, tc=5.0 * u.Gyr, gamma=1.0 / u.Gyr):
        """Sigmoidal [alpha/Fe] evolution

        The [alpha/Fe] evolves as a sigmoidal function. This is not
        meant to be physically meaningful but to reproduce the exponential
        character of the chemical evolution

        Parameters
        ----------

        start   : scalar (dex)
            [alpha/Fe] of the oldest stellar population (default=-1.5)
        end     : scalar (dex)
            [alpha/Fe] of the youngest stellar population (default=0.2)
        tc      : scalar (Gyr)
            Characteristic transition time (default=5)
        gamma   : scalar
            Transition slope (default=1)

        Returns
        -------
        None
            Updates the SFH parameters of the instance
        """
        self.alpha = self._sigmoid(self.time, start, end, tc, gamma)

    def linear_met(self, start=-1.5, end=0.2, t_start=5.0 * u.Gyr, t_end=1.0 * u.Gyr):
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
        self.met = self._linear(self.time, start, end, t_start, t_end)

    def linear_alpha(self, start=-1.5, end=0.2, t_start=5.0 * u.Gyr, t_end=1.0 * u.Gyr):
        """Linear [alpha/Fe] evolution

        The [alpha/Fe] evolves as a ReLU function, i.e., constant at the beginning
        and linearly varing afterwards.

        Parameters
        ----------

        start   : scalar (dex)
            [alpha/Fe] of the oldest stellar population (default=-1.5)
        end     : scalar (dex)
            [alpha/Fe] of the youngest stellar population (default=0.2)
        t_start : scalar (Gyr)
            Start of the [alpha/Fe] variation (default=5)
        t_end : scalar (Gyr)
            End of the [alpha/Fe] variation (default=1)
        Returns
        -------
        None
            Updates the SFH parameters of the instance
        """
        self.alpha = self._linear(self.time, start, end, t_start, t_end)

    def linear_imf(self, start=-1.5, end=0.2, t_start=5.0 * u.Gyr, t_end=1.0 * u.Gyr):
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
        self.imf = self._linear(self.time, start, end, t_start, t_end)

    def sigmoid_imf(self, start=0.5, end=3.0, tc=5.0 * u.Gyr, gamma=1.0 / u.Gyr):
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
        self.imf = self._sigmoid(self.time, start, end, tc, gamma)
