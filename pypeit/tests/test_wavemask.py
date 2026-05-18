from pathlib import Path

from IPython import embed
import pytest

import numpy as np

# Run plotting in the background
import matplotlib
matplotlib.use('agg')  

from astropy.io import ascii

from pypeit.core import wavemask
from pypeit import dataPaths
from pypeit import PypeItError


def test_read():
    files = ['hydrogen.toml', 'helium.toml', 'telluric.toml', 'atm.toml']

    # Test failure when file doesn't exist
    with pytest.raises(PypeItError):
        regions = wavemask.read_wavelength_masks('does_not_exist.toml')

    # Try reading all of them
    regions = wavemask.read_wavelength_masks(files)
    assert regions.shape[0] == 45, 'Total number of regions changed'

    # Limit to just the hydrogen lines, tests passing a Path object
    regions = wavemask.read_wavelength_masks(files[0])
    assert regions.shape[0] == 35, 'Total number of hydrogen regions changed'

    # Also test passing a string
    regions = wavemask.read_wavelength_masks(str(files[0]))
    assert regions.shape[0] == 35, 'Total number of hydrogen regions changed'

    # Limit to the 'balmer' table
    regions = wavemask.read_wavelength_masks(files, tables='balmer')
    assert regions.shape[0] == 7, 'Number of Balmer lines changed'

    # Limit to the 2 tables across multiple files
    regions = wavemask.read_wavelength_masks(files, tables=['balmer', 'atm'])
    assert regions.shape[0] == 8, 'Number of Balmer lines changed'

    # This will issue a warning
    files = dataPaths.tests.get_file_path('test_mask.toml')
    regions = wavemask.read_wavelength_masks(files, tables='ignored_range')
    assert regions.shape[0] == 1, 'Should remove region with two Nones'
    assert regions[0,1] is None, 'Second element should be None'

    # Test failure when width is not provided
    with pytest.raises(PypeItError):
        regions = wavemask.read_wavelength_masks(files, tables='no_width')

    # Test failure when width is a list
    with pytest.raises(PypeItError):
        regions = wavemask.read_wavelength_masks(files, tables='width_list')


def test_mask():

    files = [
        dataPaths.masks.get_file_path('hydrogen.toml'),
        dataPaths.masks.get_file_path('atm.toml'),
    ]
    regions = wavemask.read_wavelength_masks(files)
    wave = np.arange(2900.0, 6600.0, 2.0)

    gpm = wavemask.build_wavelength_gpm(wave, regions)
    indx = wave < 3000.
    assert not np.any(gpm[indx]), 'All pixel blueward of 3000 A should be masked'

    halpha = 6564.6
    indx = np.argmin(np.absolute(wave - halpha))

    assert not gpm[indx], 'H-alpha should be masked'


def test_telluric_mask():

    # Test basic functionality
    mask_regions = wavemask.telluric_mask(0.9)

    # Read in the atmospheric transmission spectrum
    _tspec = ascii.read(dataPaths.skisim.get_file_path('mktrans_zm_10_10.dat'))
    tspec_wave = _tspec['wave'] * 10000.0
    tspec_tran = _tspec['trans']
    tspec_sres = None

    # Should fail on a junk file
    with pytest.raises(PypeItError):
        mask_regions = wavemask.telluric_mask(0.9, file='junk.dat')

    # Test failure if tspec tuple doesn't have three elements
    with pytest.raises(PypeItError):
        mask_regions = wavemask.telluric_mask(0.9, tspec=(tspec_wave, tspec_tran))

    # Should be identical to reading from the file
    _mask_regions = wavemask.telluric_mask(0.9, tspec=(tspec_wave, tspec_tran, tspec_sres))
    assert np.array_equal(mask_regions, _mask_regions), 'Difference between direct and from file'

    # Should be fewer regions when downgrading the resolution
    _mask_regions = wavemask.telluric_mask(0.9, sres=2000.)
    assert mask_regions.shape[0] > _mask_regions.shape[0], \
        'Should be fewer regions at lower resolution'
    assert _mask_regions.shape[0] == 140, 'Number of regions changed'

    # Mask as set of provided wavelengths
    wave = np.arange(9000.0, 24000.0, 1.0)
    gpm = wavemask.telluric_mask(0.9, wave=wave, sres=2000.)
    assert wave.size == gpm.size, 'Should return a vector that matches the wavelength vector'

    # Return the regions instead
    _mask_regions = wavemask.telluric_mask(0.9, wave=wave, sres=2000., return_regions=True)
    assert _mask_regions.ndim == 2, 'Should return a set of regions'
    assert _mask_regions.shape[0] == 30, 'Number of regions changed'
    assert np.amax(_mask_regions) < 24000, \
        'There should be no regions beyond the wavelength range of the input vector'
    
    # Try the plot with ...
    gpm = wavemask.telluric_mask(0.9, wave=wave, sres=2000., plot=True)
    # ... and without providing the wavelength vector
    mask_regions = wavemask.telluric_mask(0.9, sres=2000., plot=True)

    # Try writing the plot to a file ...
    ofile = Path().absolute() / 'test_telluric_mask_qa.png'
    if ofile.is_file():
        ofile.unlink()
    # ... with ...
    gpm = wavemask.telluric_mask(0.9, wave=wave, sres=2000., plot=ofile)
    assert ofile.is_file(), 'Should have written the plot to a file'
    ofile.unlink()
    # ... and without providing the wavelength vector
    mask_regions = wavemask.telluric_mask(0.9, sres=2000., plot=ofile)
    assert ofile.is_file(), 'Should have written the plot to a file'
    ofile.unlink()
    # Test passing just the string
    mask_regions = wavemask.telluric_mask(0.9, sres=2000., plot=ofile.name)
    assert ofile.is_file(), 'Should have written the plot to a file'
    ofile.unlink()

    # Test when there's no overlap
    wave = np.arange(2900.0, 6600.0, 1.0)
    gpm = wavemask.telluric_mask(0.9, wave=wave)
    assert np.all(gpm), 'All the pixels should be good because there are no overlaps'
    mask_regions = wavemask.telluric_mask(0.9, wave=wave, return_regions=True)
    assert mask_regions.shape == (0,), 'Should not find any regions to mask'

