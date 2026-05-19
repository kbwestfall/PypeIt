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


def load_spectra(inp, extract=None, fluxed=False, include_flat=False, chk_version=True):
    """
    Parse the input into a :class:`~pypeit.core.spectrum.SpectrumList` object.

    Parameters
    ----------
    specfile : :obj:`str`, :class:`Path`, :class:`~pypeit.specobjs.SpecObjs`, :class:`~pypeit.onespec.OneSpec`
        A file written by :class:`~pypeit.specobjs.SpecObjs` or
        :class:`~pypeit.onespec.OneSpec`, or an instance of one of those
        classes.
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
    include_flat : :obj:`bool`, optional
        If True, include the extracted flat spectrum as an associated array.
        Can only be true if the input is a spec1d file.  If this is True and the
        input is a onespec file, the function will raise an exception.
    chk_version : :obj:`bool`, optional
        When reading in existing files written by PypeIt, perform strict
        version checking to ensure a valid file.  If False, the code will
        try to keep going, but this may lead to faults and quiet failures.
        User beware!

    Returns
    -------
    :class:`astropy.io.fits.Header`
        The primary header of the input file.
    :class:`~pypeit.core.spectrum.SpectrumList`
        List of :class:`~pypeit.core.spectrum.Spectrum` objects.
    """

    if isinstance(inp, specobjs.SpecObjs) or specobjs.SpecObjs.is_specobjs_file(inp):
        # Parse the SpecObjs object
        if isinstance(inp, specobjs.SpecObjs):
            sobjs = inp
        else:
            # Load a spec1d file
            try:
                sobjs = specobjs.SpecObjs.from_fitsfile(inp, chk_version=chk_version)
            except PypeItError as e:
                raise PypeItError(
                    f'Reading {inp} failed, despite indications that it is a PypeIt spec1d file.  '
                    f'The original error is: {e}'
                )
        return sobjs.header, sobjs.to_spectrum(
            extract=extract, fluxed=fluxed, include_flat=include_flat
        )

    # Parse the OneSpec object
    if isinstance(inp, onespec.Onespec):
        ospec = inp
    else:
        # Load a OneSpec file
        try:
            ospec = onespec.OneSpec.from_file(inp, chk_version=chk_version)
        except PypeItError as e:
            raise PypeItError(
                f'Unable to load data in {inp} using the SpecObjs or OneSpec datamodels.  The '
                'DMODCLS keyword is either not present or not equal to SpecObjs in the primary'
                'header, and the error raised when attempting to read the file using OneSpec '
                f'was: {e}'
            )
        
    if include_flat:
        # TODO: Issue a warning instead.
        raise PypeItError(
            'Spectra read from OneSpec output files do not contain the flat spectrum.  To '
            'continue, you must set include_flat=False.'
        )

    if fluxed != ospec.fluxed:
        raise PypeItError(
            f'There is a mismatch between flux-calibration status for the provided spectrum '
            f'({ospec.fluxed}) and the requested status (fluxed={fluxed}).  Unable to proceed.'
        )

    if extract is not None and ospec.ext_mode != extract:
        raise PypeItError(
            f'There is a mismatch between the extraction used for the provided spectrum '
            f'({ospec.ext_mode}) and the requested status (extract={extract}).  Unable to proceed.'
        )
    
    return ospec.head0, spectrum.SpectrumList([ospec.to_spectrum()])


def load_standard(
    specfiles, names=None, extract=None, fluxed=False, include_flat=False, multi_spec_det=None,
    chk_version=True
):
    """

    Parameters
    ----------
    specfiles : str, Path, list
        One or more pypeit files with 1D standard-star spectra.
    names : str, list
        One or more pre-identified source names that are the standard-star
        spectra.  This is only relevant if the provided ``specfiles`` are PypeIt
        spec1d files; i.e., this is ignored for files written by
        :class:`~pypeit.onespec.Onespec`.  If multiple files are provided, a
        name should be provided for each file; however, the name can be ``None``
        if you want the code to identify the highest S/N spectrum as the
        standard star.  If more than one source name is required for a given
        file, the corresponding entry in the ``names`` list can itself be a list
        of names; see :func:`~pypeit.specobjs.SpecObjs.get_std`.
    extract : str, optional
        The extraction type to use; see :func:`~pypeit.loader.load_spectra`.
    fluxed : bool, optional
        Whether or not the loaded spectra should be flux-calibrated; see
        :func:`~pypeit.loader.load_spectra`.
    include_flat : bool, optional
        Whether or not to include the extracted flat-field spectra; see
        :func:`~pypeit.loader.load_spectra`.
    multi_spec_det : list, optional
        When automatically detecting the standard-star spectrum, load spectra
        that cross multiple detectors; see
        :func:`~pypeit.specobjs.SpecObjs.get_std`.
    chk_version : bool, optional
        Check the datamodel version of each file.

    Returns
    -------
    :class:`~pypeit.spectrum.SpectrumList`
        Standard-star spectra.
    bool
        Flag that the spectra should be spliced together
    """
    # Check the input files
    _specfiles = [specfiles] if isinstance(specfiles, (str,Path)) else specfiles
    _specfiles = [Path(s).absolute() for s in _specfiles]
    bad_files = [not s.is_file() for s in _specfiles]
    if any(bad_files):
        raise PypeItError(f'The following files do not exist: {np.asarray(_specfiles)[bad_files]}')

    # Parse the source names
    if names is None:
        _names = [None]*len(_specfiles)
    else:
        _names = names if isinstance(names, list) else [names]
    if len(_names) != len(_specfiles):
        raise PypeItError(
            f'The number of names provided ({len(_names)}) does not match the number of spectrum '
            'files.'
        )

    spec = spectrum.SpectrumList()
    dets = []

    # TODO: This effectively allows for lists that combine both spec1d files and
    # onespec files.  Is that a reasonable thing to do?
    for name, specfile in zip(_names, _specfiles):

        if specobjs.SpecObjs.is_specobjs_file(specfile):
            sobj = specobjs.SpecObjs.from_fitsfile(
                specfile, chk_version=chk_version
            ).get_std(name=name, multi_spec_det=multi_spec_det, split_mosaic=True)
            if sobj is None:
                raise PypeItError(f'Unable to read standard star spectrum from: {specfile}')
            dets += sobj.DET.tolist()
        else:
            sobj = specfile

        spec += load_spectra(
            sobj, extract=extract, fluxed=fluxed, include_flat=include_flat,
            chk_version=chk_version
        )[1]

        # TODO: Need to find an equivalent check for this.  E.g., the number of expected orders?
#        if ospec.head0['PYPELINE'] == 'Echelle':
#            raise PypeItError(
#                'Standard star 1D spectrum from OneSpec class cannot be used for Echelle data.'
#            )

    if len(spec) == 0:
        raise PypeItError(
            f'Unable to load any standard spectra from the provided file(s):  {specfiles}'
        )

    # Sort by wavelength
    srt = np.argsort(max([np.max(s.wave) for s in spec]), kind='stable')
    spec = spec[srt]

    # splice together also mosaic-reduced spectra that have been split
    splice_multi_det = len(_specfiles) > 1 or (len(dets) > 0 and np.unique(dets).size > 1)

    # TODO: Return `dets` instead of `splice_multi_det`?
    return spec, splice_multi_det
