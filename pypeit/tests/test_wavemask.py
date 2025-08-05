
from IPython import embed

import pytest

import numpy as np

from pypeit.core import wavemask
from pypeit import dataPaths
from pypeit.pypmsgs import PypeItError


def test_read():
    files = [
        dataPaths.masks.get_file_path('hydrogen.toml'),
        dataPaths.masks.get_file_path('helium.toml'),
        dataPaths.masks.get_file_path('telluric.toml'),
        dataPaths.masks.get_file_path('atm.toml'),
    ]

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
