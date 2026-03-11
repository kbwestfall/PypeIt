"""
Provides convenience functions for scripts that load spectra output by PypeIt.
"""
from pathlib import Path

from IPython import embed
import numpy as np

from pypeit import inputfiles
from pypeit import io
from pypeit import log
from pypeit import onespec
from pypeit import PypeItError
from pypeit import specobjs
from pypeit.core import spectrum
from pypeit.spectrographs.util import load_spectrograph


def get_pypeitpar(spec1dfile, ifile=None, secondary_ifile_class=None):
    """
    Provided some input files, construct the objects providing the spectrograph
    and the processing parameters.

    This function is particularly intended to support "after-burn" scripts that
    can operate on either ``spec1d_`` files (which use the
    :class:`~pypeit.specobjs.SpecObjs` datamodel) or coadding output files
    (which use the :class:`~pypeit.onespec.OneSpec` datamodel).

    Parameters
    ----------
    spec1dfile : str, Path
        An input spectrum file that the script is expected to use.  This *must*
        be a file produced by PypeIt, but it can have been written by either
        :class:`~pypeit.specobjs.SpecObjs` or :class:`~pypeit.onespec.OneSpec`.
    ifile : str, Path, optional
        A file with user-specified processing parameters.  This can be a
        ``.pypeit`` file, or another file type used to specify these parameters
        (e.g., a .tell file for running ``pypeit_tellfit`` or a .sens file for
        running ``pypeit_sensfunc``).  If the file is not a ``.pypeit`` file,
        you must provide ``secondary_ifile_class``.  If None, the default
        parameters specific to the instrument configuration determined from
        ``spec1dfile`` are used.
    secondary_ifile_class : object
        A subclass of :class:`~pypeit.inputfiles.InputFile` used to read
        ``ifile``, in the case that it is *not* a ``.pypeit`` file.  The
        function always first checks if the file can be read by 
        :class:`~pypeit.inputfiles.PypeItFile`; this class is the fallback
        option if that fails.  Ignored if ``ifile`` is None.

    Returns
    -------
    par : :class:`~pypeit.par.pypeitpar.PypeItPar`
        The full pypeit parameter set
    spec : :class:`~pypeit.spectrographs.spectrograph.Spectrograph`
        The spectrograph subclass appropriate to the provided ``spec1dfile``.
    """
    # Check the input spec1d file
    _spec1dfile = Path(spec1dfile).absolute()
    if not _spec1dfile.is_file():
        raise FileNotFoundError(f'Spec1d file not found: {_spec1dfile}')
    
    # TODO: We can add
    #   spec.dispname = hdu[0].header['DISPNAME']
    # below if needed!

    # TODO: Do we also need to build the primary header, as done in the old
    # SensFunc script?

    if ifile is None:
        # An InputFile is not provided, so only use the configuration specific parameters
        with io.fits_open(_spec1dfile) as hdu:
            spec = load_spectrograph(hdu[0].header['PYP_SPEC'], pypeit_fits=True)
            par = spec.config_specific_par(hdu)
        return par, spec

    # Check input secondary class
    if (
        secondary_ifile_class is not None
        and not issubclass(secondary_ifile_class, inputfiles.InputFile)
    ):
        raise PypeItError(
            'If providing a secondary_ifile_class, it must be a subclass of '
            f'inputfiles.InputFile; {secondary_ifile_class.__name__} is not.'
        )

    # First try to read the input file as a .pypeit file.
    try:
        pfile = inputfiles.PypeItFile.from_file(ifile)
    except PypeItError as e:
        if secondary_ifile_class is None:
            raise PypeItError(
                f'Could not read {ifile} as a .pypeit file.  Provide a secondary_ifile_class.'
            )
        log.warning(
            f'Could not read {ifile} as a .pypeit file.  Will next try '
            f'{secondary_ifile_class.__name__}.  Error was: {e}'
        )
    else:
        spec, par, _ = pfile.get_pypeitpar()
        return par, spec
    
    # The function should not get here unless the attempt to read ``ifile`` as a
    # pypeit file failed.  Attempt to use the secondary_ifile_class.  If
    # secondary_ifile_class is None, the function should have raised an
    # exception above (in the exception of the first try-except block).
    try:
        pfile = secondary_ifile_class.from_file(ifile)
    except PypeItError as e:
        raise PypeItError(
            f'Cannot parse {ifile} as a .pypeit file or {secondary_ifile_class.__name__} file!'
        )
    else:
        # NOTE: We set `pypeit_fits=True` below because, *by definition*,
        # the input files are PypeIt output files.
        with io.fits_open(_spec1dfile) as hdu:
            spec, par, _ = pfile.get_pypeitpar(
                config_specific_file=hdu, spectrograph_name=hdu[0].header['PYP_SPEC'],
                pypeit_fits=True
            )
        return par, spec


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
    # TODO: Can we use try/except blocks to avoid opening the fits file?  Is
    # that any more efficient than what's being done already?

    # Get the type of file.
    with io.fits_open(specfile) as hdu:
        is_specobjs = 'DMODCLS' in hdu[0].header and hdu[0].header['DMODCLS'] == 'SpecObjs'

    # Load a spec1d file
    if is_specobjs:
        sobjs = specobjs.SpecObjs.from_fitsfile(specfile, chk_version=chk_version)
        return sobjs.header, specobjs_to_spectrum(sobjs, extract=extract, fluxed=fluxed)

    # Load a OneSpec file
    try:
        spec = onespec.OneSpec.from_file(specfile, chk_version=chk_version)
    except PypeItError as e:
        # TODO: Catch a specific exception!
        raise PypeItError(
            f'Unable to load data in {specfile} using the SpecObjs or OneSpec datamodels.  The '
            'DMODCLS keyword is either not present or not equal to SpecObjs in the primary '
            f'header, and the error raised when attempting to read the file using OneSpec was: {e}'
        )

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

    return spec.head0, [
        spectrum.Spectrum(_wave, spec.flux, ivar=spec.ivar, gpm=_gpm, meta=spec.spect_meta)
    ]


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

