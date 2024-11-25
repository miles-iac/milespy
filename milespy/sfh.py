# -*- coding: utf-8 -*-
import logging
from typing import Optional

import numpy as np
from astropy import units as u
from scipy.integrate import trapezoid

logger = logging.getLogger("milespy.sfh")

DEFAULT_NBINS = 20


class SFH:
    """
    Class for manipulating star formation histories (SFH) and create derived spectra

    Attributes
    ----------
    time: ~astropy.units.Quantity
        Look-back time
    sfr: ~astropy.units.Quantity
        Values of the star formation rate (SFR) at each time
    met: ~astropy.units.Quantity
        Values of the metallicity at each time (dex)
    alpha: ~astropy.units.Quantity
        Values of [alpha/Fe] at each time (dex)
    imf: ~astropy.units.Quantity
        Values of the IMF slope
    """

    @u.quantity_input
    def __init__(
        self, time: u.Quantity[u.Gyr] = np.linspace(0.035, 13.5, DEFAULT_NBINS) << u.Gyr
    ):
        """
        Create a base SFH object

        Parameters
        ----------
        time: ~astropy.units.Quantity
            Look-back time array for the SFH samples
        """
        nbins = len(time)
        self.time = time
        self.sfr = np.zeros(nbins) << u.Msun / u.Gyr
        self.met = np.zeros(nbins) << u.dex
        self.alpha = np.zeros(nbins) << u.dex
        self.imf = np.full(nbins, 1.3) << u.dimensionless_unscaled

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

    def _compute_time_weights(self):
        dt = self.time - np.roll(self.time, 1)
        dt[0] = dt[1]
        self.time_weights = (self.sfr * dt).to(u.Msun)

    def _normalize(self, sfr):
        norm = trapezoid(sfr, x=self.time.to_value(u.yr))
        self.sfr = sfr / norm * self.mass / u.yr

        self._compute_time_weights()

    @u.quantity_input
    def sfr_tau(
        self,
        start: u.Quantity[u.Gyr] = 10.0 * u.Gyr,
        tau: u.Quantity[u.Gyr] = 1.0 * u.Gyr,
        mass: u.Quantity[u.Msun] = 1.0 * u.Msun,
    ):
        r"""Exponentially declining SFR

        This is a standard tau model where the SFR is given by

        .. math::

            \text{SFR}(t) =
            \begin{cases}
            0 & \text{if}\; t < t_0 \\
            e^{- (t-t_0)/\tau } & \text{if}\; t >= t_0 \\
            \end{cases}


        Parameters
        ----------

        start : ~astropy.units.Quantity
            Start of the burst (default=10 Gyr)
        tau   : ~astropy.units.Quantity
            e-folding time (default=1 Gyr)
        mass : ~astropy.units.Quantity
            Total formed mass (default=1 Msun)
        """
        for inp in (start, tau, mass):
            self._validate_scalar(inp)

        # Exponentially declining SFR
        self.time <= start
        sfr = np.zeros(self.time.shape)
        sfr[(self.time <= start)] = np.exp(
            -(start - self.time[(self.time <= start)]) / tau
        )
        self.mass = mass
        self._normalize(sfr)

    @u.quantity_input
    def sfr_delayed_tau(
        self,
        start: u.Quantity[u.Gyr] = 10.0 * u.Gyr,
        tau: u.Quantity[u.Gyr] = 1.0 * u.Gyr,
        mass: u.Quantity[u.Msun] = 1.0 * u.Msun,
    ):
        r"""Delayed exponentially declining SFR

        This is a standard tau model where the SFR is given by

        .. math::

            \text{SFR}(t) =
            \begin{cases}
            0 & \text{if}\; t < t_0 \\
            (t_0-t)e^{- (t-t_0)/\tau } & \text{if}\; t >= t_0 \\
            \end{cases}

        Parameters
        ----------

        start : ~astropy.units.Quantity
            Start of the burst (default=10 Gyr)
        tau   : ~astropy.units.Quantity
            e-folding time (default=1 Gyr)
        mass : ~astropy.units.Quantity
            Total formed mass (default=1 Msun)

        """

        for inp in (start, tau):
            self._validate_scalar(inp)

        sfr = np.zeros(self.time.shape)
        sfr[(self.time <= start)] = (start - self.time[(self.time <= start)]) * np.exp(
            -(start - self.time[(self.time <= start)]) / tau
        )

        self.mass = mass
        self._normalize(sfr)

    @u.quantity_input
    def sfr_lognormal(
        self,
        Tpeak: u.Quantity[u.Gyr] = 10.0 * u.Gyr,
        tau: u.Quantity[u.Gyr] = 1.0 * u.Gyr,
        mass: u.Quantity[u.Msun] = 1.0 * u.Msun,
    ):
        r"""Lognormal SFR

        The time evolution of the SFR is given by

        .. math::

            \text{SFR}(t) = e^{ -(t_0- \log(t_n))^2 / 2\tau^2} / t_n


        Note that :math:`t_n` is in this case the time since the Big Bang and not
        lookback time as in the SSP. See details in Diemer et al. 2017,
        ApJ, 839, 26, Appendix A.1


        Parameters
        ----------

        Tpeak : ~astropy.units.Quantity
            Time of the SFR peak (default=10 Gyr)
        tau   : ~astropy.units.Quantity
            Characteristic time-scale (default=1 Gyr)
        mass : ~astropy.units.Quantity
            Total formed mass (default=1 Msun)
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

        self._normalize(sfr)

    @u.quantity_input
    def sfr_double_power_law(
        self,
        a=5,
        b=5,
        tp: u.Quantity[u.Gyr] = 10 * u.Gyr,
        mass: u.Quantity[u.Msun] = 1.0 * u.Msun,
    ):
        r"""Double power law SFR evolution

        The SFR as a function of time is given by (Behroozi et al. 2013)

        .. math::

            \text{SFR}(t_n) = ((t_n/t_p)^a + (t_n/t_p)^{-b})^{-1}

        As for the lognormal SFR, :math:`t_n` refers to time since the Big Bang


        Parameters
        ----------
        a     : float
            falling slope (default=5)
        b     : float
            rising slope (default=5)
        tp   : ~astropy.units.Quantity
            similar to the SFR peak in look-back time (default=10 Gyr)
        mass : ~astropy.units.Quantity
            Total formed mass (default=1 Msun)
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

    @u.quantity_input
    def sfr_custom(
        self, sfr: u.Quantity[u.Msun / u.yr], mass: Optional[u.Quantity[u.Msun]] = None
    ):
        """User-defined SFR

        Parameters
        ----------
        sfr     : ~astropy.units.Quantity
            Star formation rate array, with the same shape as `time`.
        mass     : ~astropy.units.Quantity
            If given, normalize the input SFR such that it produces this mass.

        """
        if mass is None:
            self._set_input_sfr(sfr)
        else:
            self.mass = mass
            self._normalize(sfr)

    @u.quantity_input
    def _set_input_sfr(self, sfr: u.Quantity[u.Msun / u.Gyr]):
        self.mass = trapezoid(sfr, x=self.time)
        self.sfr = sfr
        self._compute_time_weights()

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

    @u.quantity_input
    def met_sigmoid(
        self,
        start: u.Quantity[u.dex] = -1.5,
        end: u.Quantity[u.dex] = 0.2,
        tc: u.Quantity[u.Gyr] = 5.0 * u.Gyr,
        gamma: u.Quantity[1 / u.Gyr] = 1.0 / u.Gyr,
    ):
        """Sigmoidal metallicity evolution

        The metallicity evolves as a sigmoidal function. This is not
        meant to be physically meaningful but to reproduce the exponential
        character of the chemical evolution

        Parameters
        ----------

        start   : ~astropy.units.Quantity (dex)
            Metallicity of the oldest stellar population (default=-1.5 dex)
        end     : ~astropy.units.Quantity (dex)
            Metallicity of the youngest stellar population (default=0.2 dex)
        tc      : ~astropy.units.Quantity (Gyr)
            Characteristic transition time (default=5 Gyr)
        gamma   : ~astropy.units.Quantity
            Transition slope (default=1/Gyr)

        """
        self.met = self._sigmoid(self.time, start, end, tc, gamma)

    @u.quantity_input
    def alpha_sigmoid(
        self,
        start: u.Quantity[u.dex] = 0.4,
        end: u.Quantity[u.dex] = 0.0,
        tc: u.Quantity[u.Gyr] = 5.0 * u.Gyr,
        gamma: u.Quantity[1 / u.Gyr] = 1.0 / u.Gyr,
    ):
        """Sigmoidal [alpha/Fe] evolution

        The [alpha/Fe] evolves as a sigmoidal function. This is not
        meant to be physically meaningful but to reproduce the exponential
        character of the chemical evolution

        Parameters
        ----------

        start   : ~astropy.units.Quantity (dex)
            [alpha/Fe] of the oldest stellar population (default=-1.5 dex)
        end     : ~astropy.units.Quantity (dex)
            [alpha/Fe] of the youngest stellar population (default=0.2 dex)
        tc      : ~astropy.units.Quantity (Gyr)
            Characteristic transition time (default=5 Gyr)
        gamma   : ~astropy.units.Quantity
            Transition slope (default=1/Gyr)

        """
        self.alpha = self._sigmoid(self.time, start, end, tc, gamma)

    @u.quantity_input
    def met_linear(
        self,
        start: u.Quantity[u.dex] = -1.5,
        end: u.Quantity[u.dex] = 0.2,
        t_start: u.Quantity[u.Gyr] = 5.0 * u.Gyr,
        t_end: u.Quantity[u.Gyr] = 1.0 * u.Gyr,
    ):
        """Linear metallicity evolution

        The metallicity evolves as a ReLU function, i.e., constant at the beginning
        and linearly varing afterwards.

        Parameters
        ----------

        start   : ~astropy.units.Quantity (dex)
            Metallicity of the oldest stellar population (default=-1.5 dex)
        end     : ~astropy.units.Quantity (dex)
            Metallicity of the youngest stellar population (default=0.2 dex)
        t_start : ~astropy.units.Quantity (Gyr)
            Start of the metallicity variation (default=5 Gyr)
        t_end : ~astropy.units.Quantity (Gyr)
            End of the metallicity variation (default=5 Gyr)
        """
        self.met = self._linear(self.time, start, end, t_start, t_end)

    @u.quantity_input
    def alpha_linear(
        self,
        start: u.Quantity[u.dex] = -1.5,
        end: u.Quantity[u.dex] = 0.2,
        t_start: u.Quantity[u.Gyr] = 5.0 * u.Gyr,
        t_end: u.Quantity[u.Gyr] = 1.0 * u.Gyr,
    ):
        """Linear [alpha/Fe] evolution

        The [alpha/Fe] evolves as a ReLU function, i.e., constant at the beginning
        and linearly varing afterwards.

        Parameters
        ----------

        start   : ~astropy.units.Quantity (dex)
            [alpha/Fe] of the oldest stellar population (default=-1.5 dex)
        end     : ~astropy.units.Quantity (dex)
            [alpha/Fe] of the youngest stellar population (default=0.2 dex)
        t_start : ~astropy.units.Quantity (Gyr)
            Start of the [alpha/Fe] variation (default=5 Gyr)
        t_end : ~astropy.units.Quantity (Gyr)
            End of the [alpha/Fe] variation (default=1 Gyr)
        """
        self.alpha = self._linear(self.time, start, end, t_start, t_end)

    @u.quantity_input
    def imf_linear(
        self,
        start: u.Quantity = -1.5,
        end: u.Quantity = 0.2,
        t_start: u.Quantity[u.Gyr] = 5.0 * u.Gyr,
        t_end: u.Quantity[u.Gyr] = 1.0 * u.Gyr,
    ):
        """Linear IMF slope evolution

        The IMF slope evolves as a ReLU function, i.e., constant at the beginning
        and linearly varing afterwards.

        Parameters
        ----------

        start   : ~astropy.units.Quantity
            IMF of the oldest stellar population (default=-1.5)
        end     : ~astropy.units.Quantity
            IMF  of the youngest stellar population (default=0.2)
        t_start : ~astropy.units.Quantity (Gyr)
            Start of the IMF variation (default=5 Gyr)
        t_end : ~astropy.units.Quantity (Gyr)
            End of the IMF variation (default=1 Gyr)
        """
        self.imf = self._linear(self.time, start, end, t_start, t_end)

    @u.quantity_input
    def imf_sigmoid(
        self,
        start: u.Quantity = 0.5,
        end: u.Quantity = 3.0,
        tc: u.Quantity[u.Gyr] = 5.0 * u.Gyr,
        gamma: u.Quantity[1 / u.Gyr] = 1.0 / u.Gyr,
    ):
        """Sigmoidal IMF slope evolution

        The IMF slope  evolves as a sigmoidal function. This is not
        meant to be physically meaningful but to track the chemical
        variations (see e.g. Martin-Navarro et al. 2015)

        Parameters
        ----------

        start   : ~astropy.units.Quantity (dex)
            IMF slope  of the oldest stellar population (default=0.5)
        end     : ~astropy.units.Quantity (dex)
            IMF slope  of the youngest stellar population (default=3.0)
        tc      : ~astropy.units.Quantity (Gyr)
            Characteristic transition time (default=5 Gyr)
        gamma   : ~astropy.units.Quantity
            Transition slope (default=1 Gyr)

        """
        self.imf = self._sigmoid(self.time, start, end, tc, gamma)
