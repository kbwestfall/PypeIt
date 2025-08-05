"""
Temporary light-weight spectrum object.

Ideally this would be replaced by specutils.Spectrum

"""

import numpy as np

from pypeit import msgs


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
        if self.flux.size != self.wave.size:
            msgs.error('Wavelength and flux arrays do not have the same size.')

        if ivar is None:
            self.ivar = None
        else:
            self.ivar = np.asarray(ivar, dtype=float)
            if self.ivar is not None and self.ivar.size != self.wave.size:
                msgs.error('Wavelength and inverse variance arrays do not have the same size.')

        if gpm is None:
            self.gpm = np.ones(wave.size, dtype=bool)
        else:
            self.gpm = np.asarray(gpm, dtype=bool)
            if self.gpm.size != self.wave.size:
                msgs.error('Wavelength and good-pixel arrays do not have the same size.')

        self.meta = meta

    @property
    def size(self):
        return self.wave.size

