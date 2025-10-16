"""
Basic utility for defining wavelength masks.
"""

import tomllib
from pathlib import Path

from IPython import embed

import numpy as np
from matplotlib import pyplot

from astropy.io import ascii

from pypeit import dataPaths
from pypeit import log
from pypeit import PypeItError
from pypeit import utils
from pypeit.sampling import Resample
from pypeit.wavemodel import conv2res


def read_wavelength_masks(files, tables=None):
    r"""
    Read one or more TOML files with wavelength mask definitions.

    Parameters
    ----------
    files : str, Path, list, `numpy.ndarray`_
        One or more files to read
    tables : str, list, optional
        Restrict the masks to one or more tables in the set of files provided

    Returns
    -------
    `numpy.ndarray`_
        An array with size :math:`(N_{\rm mask},2)`, where :math:`N_{\rm mask}`
        is the number of mask regions with a starting and ending wavelength.
    """
    _files = [files] if isinstance(files, (str, Path)) else files
    _files = [dataPaths.masks.get_file_path(f) for f in _files]

    if tables is None:
        _tables = None
    else:
        _tables = [tables] if isinstance(tables, str) else tables
    regions = []
    mask_keys = []
    valid_set_keys = set(['range', 'center_width', 'center', 'width'])
    for f in _files:
        _f = Path(f).absolute()
        if not _f.is_file():
            raise PypeItError(f'{f} does not exist!')
        with open(_f, 'rb') as inp:
            data = tomllib.load(inp)
        for key in data.keys():
            if _tables is not None and key not in _tables:
                continue
            # Check if the mask set has already been read
            if key in mask_keys:
                log.warning(f'{key} mask set in {f} already parsed by previous file.  '
                          'Concatenated masks may have repeated regions.')
            else:
                mask_keys += [key]

            # Check the keys used to define the mask
            unknown_keys = set(data[key].keys()) - valid_set_keys
            if len(unknown_keys) > 0:
                log.warning(f'Valid mask keys are {valid_set_keys}; ignoring {unknown_keys}')

            # Generate the regions to mask    
            if 'range' in data[key].keys():
                regions += data[key]['range']
            if 'center_width' in data[key].keys():
                regions += [[c-w/2, c+w/2] for c,w in data[key]['center_width']]
            if 'center' in data[key].keys():
                if 'width' not in data[key].keys():
                    raise PypeItError('When using center keyword, must also provide width key.')
                if isinstance(data[key]['width'], list):
                    raise PypeItError('When using center keyword, width must be a single value.')
                regions += [
                    [c-data[key]['width']/2, c+data[key]['width']/2] for c in data[key]['center']
                ]

    # Convert to an array and set any None strings to None type
    regions = np.asarray(regions, dtype=object)
    indx = regions == 'None'
    if not np.any(indx):
        return regions

    # Convert the strings to booleans
    regions[indx] = None

    # Make sure that all of the regions have a least one edge that is not
    # None.
    both_none = np.all(indx, axis=1)
    if not np.any(both_none):
        return regions

    log.warning('Ignoring ranges with but the lower and upper limits were set to None.')
    return regions[np.logical_not(both_none),:]


def build_wavelength_gpm(wave, regions):
    r"""
    Construct a good-pixel mask for a wavelength vector based on a set of
    regions that should be masked.

    Parameters
    ----------
    wave : `numpy.ndarray`_
        Wavelength array.  Can have any shape.  Units must match the ``regions``
        array.
    regions : array-like
        Wavelength regions to mask.  Units must match the ``wave`` array.  Shape
        must be :math:`(N_{\rm mask},2)`, where :math:`N_{\rm mask}` is the
        number of mask regions with a starting and ending wavelength.  Starting
        and ending regions can be ``None``, meaning that the region only has a
        upper or lower boundary; e.g., a mask range of ``[None, 3100.0]`` means
        mask all wavelengths less than 3100.  See
        :func:`~pypeit.core.wavemask.read_wavelength_masks`.

    Returns
    -------
    `numpy.ndarray`_
        Boolean array with the same shape as ``wave``.  Values are True for
        wavelengths that are *not* within the wavelength ranges provided by
        ``regions``.
    """
    # Check input
    _regions = np.asarray(regions)
    if _regions.ndim != 2:
        raise PypeItError('regions array must be 2D')
    if _regions.shape[1] != 2:
        raise PypeItError('regions array must have 2 elements in the 2nd dimension')

    # Create and edit the mask
    gpm = np.ones(wave.shape, dtype=bool)
    for r in _regions:
        if r[0] is None and r[1] is None:
            continue

        if r[0] is None:
            _gpm = wave > r[1]
        elif r[1] is None:
            _gpm = wave < r[0]
        else:
            _gpm = (wave < r[0]) | (wave > r[1])
        gpm &= _gpm
    return gpm


def telluric_mask(threshold, wave=None, sres=None, file='mktrans_zm_10_10.dat', tspec=None,
                  return_regions=False, plot=False):
    r"""
    Construct a mask based on a minimum atmospheric transmission threshold.

    Parameters
    ----------
    threshold : float
        Transmission threshold.  Regions below this threshold are masked.
    wave : array-like, optional
        Wavelengths of the observed spectrum in angstrom.  If provided and
        ``return_regions`` is False, a good-pixel mask is returned, identifying
        wavelengths at which the atmospheric transmission spectrum is above the
        provided threshold.  Also, if ``wave`` is provided and ``sres`` is not,
        the wavelength vector is used to approximate the spectral resolution;
        see ``sres``.  If None, the mask is always returned as a set of
        wavelength regions.
    sres : array-like, optional
        The spectral resolution of the *observed* spectrum to be masked.  If
        both ``wave`` and ``sres`` are None, the mask is returned based on the
        native resolution of the transmission spectrum.  Otherwise, the mask is
        returned at the resolution of the observed data.  This argument provides
        the spectral resolution (:math:`R = \lambda/\Delta\lambda`) directly;
        however, if ``wave`` is provided and ``sres`` is None, the observed
        spectral resolution is estimated assuming a resolution element of 3
        pixels.
    file : str, optional
        A file with the transmission spectrum.  The file should have two
        columns: (1) wavelength in micron and (2) transmission fraction.  This
        can be a local file or a file distributed by PypeIt.  The transmission
        spectrum can be provided directly using ``tspec``.
    tspec : tuple, optional
        A tuple with three array-like objects, providing the wavelengths,
        transmission fraction, and spectral resolution of the atmospheric
        transmission spectrum.  The resolution can be set to None, which means
        the resolution is ignored when convolving the atmospheric transmission
        spectrum to match the resolution of the observed data.
    plot : bool, str, Path, optional
        Show a diagnostic plot illustrating the regions to mask against the
        transmission spectrum.  If this is True, the plot is shown in a
        matplotlib window.  If a string or Path object, the plot is written to
        the provided file name.

    Returns
    -------
    `numpy.ndarray`_
        If ``wave`` was provided, this is a boolean good-pixel mask indication
        the wavelengths of regions above the provided transmission threshold.
        If ``wave`` is not provided, this is an array with shape :math:`(N_{\rm
        mask},2)`, where :math:`N_{\rm mask}` is the number of mask regions with
        a starting and ending wavelength.
    """
    # Get the transmission spectrum
    if tspec is None:
        _file = dataPaths.skisim.get_file_path(file)
        _tspec = ascii.read(_file)
        tspec_wave = _tspec['wave'] * 10000.0
        tspec_tran = _tspec['trans']
        tspec_sres = None
    else:
        if len(tspec) != 3:
            raise PypeItError('`tspec` argument must be a tuple with three elements')
        tspec_wave = np.asarray(tspec[0])
        tspec_tran = np.asarray(tspec[1])
        tspec_sres = None if tspec[2] is None else np.asarray(tspec[2])

    # Set the observed spectral resolution
    if sres is None and wave is not None:
        _sres = np.median(wave / np.median(np.diff(wave)) / 3.)
    else:
        _sres = sres

    # Get the matched-resolution telluric spectrum
    if _sres is None:
        tspec_tran_cnv = tspec_tran
    else:
        tspec_tran_cnv = conv2res(tspec_wave, tspec_tran, _sres, current_resolution=tspec_sres)[0]

    # keyword argument used for plotting
    kwargs = {'ofile': plot} if isinstance(plot, (str, Path)) else {}

    # Get the mask
    if wave is None:
        # The observed wavelength vector wasn't provides, so return a set of
        # wavelength regions
        reg_slices = utils.contiguous_true(tspec_tran_cnv < threshold)
        mask_regions = np.array([tspec_wave[reg][[0,-1]].tolist() for reg in reg_slices])
        if plot:    # Test works of plot is anything other than False or None
            telluric_mask_plot(threshold, tspec_wave, tspec_tran_cnv, mask_regions, **kwargs)
        return mask_regions

    # Resample the transmission spectrum to the observed wavelength range
    r = Resample(tspec_tran_cnv, x=tspec_wave, newx=wave, conserve=False)
    within_wave_range = r.outf > 0
    if not np.any(within_wave_range):
        log.warning('Transmission spectrum does not overlap with observed spectrum.')
        return np.array([]) if return_regions else np.ones(wave.size, dtype=bool)

    if not np.all(within_wave_range):
        log.warning('Regions of the observed spectrum that do not overlap with the transmission '
                  'spectrum will not be masked.')
    
    bpm = (r.outf > 0) & (r.outy < threshold)
    if return_regions or plot:
        reg_slices = utils.contiguous_true(bpm)
        mask_regions = np.array([r.outx[reg][[0,-1]].tolist() for reg in reg_slices])
    if plot:    # Check works if plot is anything other than False or None
        telluric_mask_plot(threshold, r.outx, r.outy, mask_regions, **kwargs)
    return mask_regions if return_regions else np.logical_not(bpm)


def telluric_mask_plot(threshold, wave, tspec_tran_cnv, mask_regions, ofile=None):
    r"""
    Diagnostic plot for masking based on the atmospheric transmission spectrum.

    Parameters
    ----------
    threshold : float
        Transmission threshold.  Regions below this threshold are masked.
    wave : `numpy.ndarray`_
        Wavelength vector
    tspec_tran_cnv : `numpy.ndarray`_
        The atmospheric transmission spectrum used to define the mask regions.
    mask_region : `numpy.ndarray`_
        Wavelength regions to mask.  Units must match the ``wave`` array.  Shape
        must be :math:`(N_{\rm mask},2)`, where :math:`N_{\rm mask}` is the
        number of mask regions with a starting and ending wavelength.  Starting
        and ending regions can be ``None``, meaning that the region only has a
        upper or lower boundary; e.g., a mask range of ``[None, 3100.0]`` means
        mask all wavelengths less than 3100.  See
        :func:`~pypeit.core.wavemask.read_wavelength_masks`.
    ofile : str, Path, optional
        Filename for the plot, if an output file is desired.  If None, the plot
        is shown to the screen.
    """
    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(2*w,h))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(wave, tspec_tran_cnv, color='k', zorder=3)
    ax.axhline(threshold, color='C3', ls='--', zorder=4)

    for reg in mask_regions:
        ax.axvspan(*reg, alpha=0.2, color='k', zorder=1)

    ax.set_xlabel('Wavelength [Angstrom]')
    ax.set_ylabel('Fractional Transmission')

    if ofile is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(ofile, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)



