"""
Basic utility for defining wavelength masks.
"""

import tomllib
from pathlib import Path

import numpy as np

from pypeit import msgs


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
            msgs.error(f'{f} does not exist!')
        with open(_f, 'rb') as inp:
            data = tomllib.load(inp)
        for key in data.keys():
            if _tables is not None and key not in _tables:
                continue
            # Check if the mask set has already been read
            if key in mask_keys:
                msgs.warn(f'{key} mask set in {f} already parsed by previous file.  '
                          'Concatenated masks may have repeated regions.')
            else:
                mask_keys += [key]

            # Check the keys used to define the mask
            unknown_keys = set(data[key].keys()) - valid_set_keys
            if len(unknown_keys) > 0:
                msgs.warn(f'Valid mask keys are {valid_set_keys}; ignoring {unknown_keys}')

            # Generate the regions to mask    
            if 'range' in data[key].keys():
                regions += data[key]['range']
            if 'center_width' in data[key].keys():
                regions += [[c-w/2, c+w/2] for c,w in data[key]['center_width']]
            if 'center' in data[key].keys():
                if 'width' not in data[key].keys():
                    msgs.error('When using center keyword, must also provide width key.')
                if isinstance(data[key]['width'], list):
                    msgs.error('When using center keyword, width must be a single value.')
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

    msgs.warn('Ignoring ranges with but the lower and upper limits were set to None.')
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
        msgs.error('regions array must be 2D')
    if _regions.shape[1] != 2:
        msgs.error('regions array must have 2 elements in the 2nd dimension')

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


