"""
A temporary module to load spectral data.

This should be replaced by some I/O infrastructure for all the various spectral output data.
"""
from IPython import embed
import numpy as np

from pypeit import io
from pypeit import onespec
from pypeit import PypeItError
from pypeit import specobjs
from pypeit.core import spectrum
from pypeit.spectrographs.util import load_spectrograph


def load_spectra(specfile, extract=None, fluxed=False, chk_version=True):
    """
    Read a spec1d or onespec file.

    Parameters
    ----------
    specfile : :obj:`str`
        File with the data
    extract : str, optional
        The extraction used to produce the spectrum.  Must be either None,
        ``'BOX'`` (for a boxcar extraction), or ``'OPT'`` for optimal
        extraction.  If None, the optimal extraction will be returned, if it
        exists, otherwise the boxcar extraction will be returned.  Only relevant
        if the input is a spec1d file.
    fluxed : bool, optional
        If True, return the flux-calibrated spectrum, if it exists.  If the flux
        calibration hasn't been performed or ``fluxed=False``, the spectrum is
        returned in counts.  Only relevant if the input is a spec1d file.
    chk_version : :obj:`bool`, optional
        When reading in existing files written by PypeIt, perform strict
        version checking to ensure a valid file.  If False, the code will
        try to keep going, but this may lead to faults and quiet failures.
        User beware!

    Returns
    -------
    :class:`astropy.io.fits.Header`:
        The primary header of the input file.
    list:
        List of :class:`~pypeit.core.spectrum.Spectrum` objects.
    """

    # TODO: Can we use try/except blocks to avoid opening the fits file?

    # Load a OneSpec file
    hdu = io.fits_open(specfile)
    is_onespec = 'DMODCLS' in hdu[1].header and hdu[1].header['DMODCLS'] == 'OneSpec'
    hdu.close()
    if is_onespec:
        spec = onespec.OneSpec.from_file(specfile, chk_version=chk_version)

        # TODO: I'm not sure which wavelength vector to use, but allowing data
        # with wave==0 wreaks havoc later on, so I deal with it here.
        _wave = spec.wave
        _gpm = spec.mask.astype(bool)
        bad_wave = _wave == 0
        if np.any(bad_wave):
            if np.any(bad_wave & _gpm):
                raise PypeItError(
                    'The input spectrum has unmasked pixels with the wavelength set to zero.'
                )
            if spec.wave_grid_mid is None:
                raise PypeItError(
                    'The input spectrum has pixels with the wavelength set to zero and no '
                    'alternative wavelength grid to use.'
                )
            _wave[bad_wave] = spec.wave_grid_mid[bad_wave]

        return spec.head0, [spectrum.Spectrum(
            _wave, spec.flux, ivar=spec.ivar, gpm=_gpm, meta=spec.spect_meta
        )]

    # Load a spec1d file
    sobjs = specobjs.SpecObjs.from_fitsfile(specfile, chk_version=chk_version)
    return sobjs.header, specobjs_to_spectrum(sobjs, extract=extract, fluxed=fluxed)


# TODO: This could possibly be a member function of pypeit.specobjs.SpecObjs.
def specobjs_to_spectrum(sobjs, extract=None, fluxed=False):
    """
    Utility function to convert a :class:`~pypeit.specobjs.SpecObjs` object into
    a list of :class:`~pypeit.core.spectrum.Spectrum` objects.

    Parameters
    ----------
    sobjs : :class:`~pypeit.specobjs.SpecObjs`
        Object with extracted 1D spectra
    extract : :obj:`str`, optional
        The type of extraction to use.  Options are 'OPT' for optimal extraction or
        'BOX' for boxcar extraction.
    fluxed : :obj:`bool`, optional
        If True, return the flux-calibrated spectrum.  If False, return the
        uncalibrated counts.

    Returns
    -------
    list
        List of :class:`~pypeit.core.spectrum.Spectrum` objects with the
        extracted spectra.
    """

    # Get the metadata
    meta_spec = load_spectrograph(sobjs.header['PYP_SPEC']).parse_spec_header(sobjs.header)

    # Build up the list of spectra
    spectra = []
    for sobj in sobjs:
        ext, cal = sobj.best_ext_match(extract=extract, fluxed=fluxed)
        func = sobj.get_box_ext if ext == 'BOX' else sobj.get_opt_ext
        wave, flux, ivar, gpm = func(fluxed=cal)
        # TODO: Deal with wave=0 data?
        spectra += [spectrum.Spectrum(wave, flux, ivar=ivar, gpm=gpm, meta=meta_spec)]
    return spectra
