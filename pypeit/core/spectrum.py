"""
Temporary light-weight spectrum object.

Ideally this would be replaced by specutils.Spectrum

"""

import copy
import numpy as np

from pypeit import msgs
from pypeit import utils


class Spectrum:
    """
    A light-weight container class for a spectrum.

    Parameters
    ----------
    wave : array-like
        Vacuum wavelengths in angstrom
    flux : array-like
        Flux data at each wavelength.  Shape must match ``wave``.
    ivar : array-like, optional
        Inverse variance in the flux.  Shape must match ``wave``.  If None,
        assumes uncertainties are unknown.
    gpm : array-like, optional
        Boolean good-pixel mask.  Shape must match ``wave``.  If None, all
        pixels are assumed to have valid data.
    meta : dict, optional
        Collection of relevant metadata.  Note that this is *not* copied.
    """
    def __init__(self, wave, flux, ivar=None, gpm=None, meta=None):
        
        self.wave = np.asarray(wave, dtype=float)

        self.flux = np.asarray(flux, dtype=float)
        if self.flux.shape != self.wave.shape:
            msgs.error('Wavelength and flux arrays do not have the same shape.')

        if ivar is None:
            self.ivar = None
        else:
            self.ivar = np.asarray(ivar, dtype=float)
            if self.ivar is not None and self.ivar.shape != self.wave.shape:
                msgs.error('Wavelength and inverse variance arrays do not have the same shape.')

        if gpm is None:
            self.gpm = np.ones(self.wave.shape, dtype=bool)
        else:
            self.gpm = np.asarray(gpm, dtype=bool)
            if self.gpm.shape != self.wave.shape:
                msgs.error('Wavelength and good-pixel arrays do not have the same size.')

        self.meta = meta

    @property
    def size(self):
        return self.wave.size
    
    @property
    def shape(self):
        return self.wave.shape
    
    @property
    def ndim(self):
        return self.wave.ndim
    
    def copy(self):
        """
        Make a deepcopy of the object
        """
        _ivar = None if self.ivar is None else self.ivar.copy()
        _meta = None if self.meta is None else copy.deepcopy(self.meta)
        return self.__class__(
            self.wave.copy(), self.flux.copy(), ivar=_ivar, gpm=self.gpm.copy(), meta=_meta
        )
    
    def multiply(self, a):
        """
        Multiply the spectrum by a scalar, vector, or another spectrum.

        This modifies the spectrum in place.  If uncertainties are available,
        they are propagated.  Any divisions by 0 result in an inverse variance
        of 0 and the good pixel mask is set to False.

        Parameters
        ----------
        a : scalar, array-like, :class:`pypeit.core.spectrum.Spectrum`
            Multiplicative factor.  If an array, its shape must match :attr:`flux`.
        """
        if isinstance(a, (int, np.integer, float, np.floating)):
            if a == 0.:
                msgs.warn('Multiplicative factor is 0!')
            self.flux *= a
            if self.ivar is not None:
                if np.absolute(a) > 0:
                    self.ivar /= a**2
                else:
                    self.ivar *= 0.
            return

        if isinstance(a, Spectrum):
            # NOTE: This does *not* check that the wavelength vectors are the same!
            if a.shape != self.shape:
                msgs.error(f'Shape mismatch between this spectrum ({self.shape}) and the spectrum '
                           f'to multiply by ({a.shape}).')
            sqr_err_ratio = None
            if self.ivar is not None:
                # Square of the ratio between the error and flux in this spectrum
                sqr_err_ratio = utils.inverse(self.flux**2 * self.ivar)
            if a.ivar is not None:
                # Square of the ratio between the error and flux in a
                a_sqr_err_ratio = utils.inverse(a.flux**2 * a.ivar)
                if sqr_err_ratio is None:
                    sqr_err_ratio = a_sqr_err_ratio
                else:
                    sqr_err_ratio += a_sqr_err_ratio
            self.flux *= a.flux
            if sqr_err_ratio is not None:
                sqr_err = self.flux**2 * sqr_err_ratio
                self.ivar = utils.inverse(sqr_err)
                self.gpm[np.logical_not(self.ivar > 0)] = False
            return
        
        _a = np.asarray(a)
        if _a.shape != self.flux.shape:
            msgs.error(f'Shape mismatch between spectrum flux array ({self.flux.shape}) and '
                       f'multiplicative factor array ({_a.shape}).')
        self.flux *= _a
        if self.ivar is not None:
            self.ivar *= utils.inverse(_a**2)
            self.gpm[np.logical_not(self.ivar > 0)] = False

    def inverse(self):
        """
        Replace the spectrum with its multiplicative inverse.

        This modifies the spectrum in place.  If uncertainties are available,
        they are propagated.  Any divisions by 0 result in an inverse variance
        of 0 and the good pixel mask is set to False.
        """
        if self.ivar is not None:
            self.ivar *= self.flux**4
            self.gpm[np.logical_not(self.ivar > 0)] = False
        self.flux = utils.inverse(self.flux)

    def to_magnitude(self, zeropoint=0.):
        r"""
        Convert the spectrum to magnitudes.

        For fluxes, :math:`f`, this returns

        .. math::

            m = -2.5 \log_{\rm 10} (f) + Z,

        where :math:`Z` is the provided zeropoint.

        This modifies the spectrum in place.  If uncertainties are available,
        they are propagated.  Any pixels with non-positive fluxes are masked.

        Parameters
        ----------
        zeropoint : float, optional
            The magnitude conversion zeropoint (see above)
        """
        if self.ivar is not None:
            self.ivar *= (self.flux * np.log(10) / 2.5)**2
        self.gpm[np.logical_not(self.flux > 0)] = False
        self.flux[np.logical_not(self.gpm)] = 0.
        self.flux[self.gpm] = -2.5 * np.log10(self.flux[self.gpm]) + zeropoint

