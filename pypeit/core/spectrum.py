"""
Temporary light-weight spectrum object.

Ideally this would be replaced by specutils.Spectrum

"""

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
    
    def multiply(self, a):
        """
        Multiply the spectrum by a scalar or vector.

        This modifies the spectrum in place.  If uncertainties are available,
        they are propagated; when the multiplicative factor is 0, the inverse
        variance (if available) is also set to 0.

        Parameters
        ----------
        a : scalar, array-like
            Multiplicative factor.  If an array, its shape must match :attr:`flux`.
        """
        if isinstance(a (int, np.integer, float, np.floating)):
            if a == 0.:
                msgs.warn('Multiplicative factor is 0!')
            self.flux *= a
            if self.ivar is not None:
                if np.absolute(a) > 0:
                    self.ivar /= a**2
                else:
                    self.ivar *= 0.
            return
        
        _a = np.asarray(a)
        if _a.shape != self.flux.shape:
            msgs.error(f'Shape mismatch between spectrum flux array ({self.flux.shape}) and '
                       f'multiplicative factor array ({_a.shape}).')
        self.flux *= _a
        if self.ivar is not None:
            self.ivar *= utils.inverse(_a**2)
            self.ivar[np.absolute(_a) == 0.] = 0.


