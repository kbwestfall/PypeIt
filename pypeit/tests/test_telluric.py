
from pathlib import Path
from IPython import embed

import numpy as np
import pytest

from pypeit import telluric
from pypeit import PypeItError


def test_init():
    test_file = Path('test.fits')
    if test_file.is_file():
        test_file.unlink()

    # File does not exist
    with pytest.raises(PypeItError):
        tellmod = telluric.model.PCATelluricModel(test_file)

    # File exists ...
    test_file.touch()
    # ... but load function undefined
    tellmod = telluric.model.TelluricModel(test_file)
    with pytest.raises(PypeItError):
        tellmod.load()
    test_file.unlink()

    tellmod = telluric.model.PCATelluricModel('TellPCA_3000_26000_R15000.fits')
    tellmod.load()


def test_pca():

    pca_file = 'TellPCA_3000_26000_R15000.fits'

    # Test init and load
    tellmod = telluric.model.PCATelluricModel(pca_file)
    tellmod.load()

    # Check some basics
    assert tellmod.wave_grid.size == 82398, 'Number of wavelengths changed'
    assert tellmod.tell_grid.shape[0] == 11, 'Number of components changed'

    # Test a restricted wavelength range
    _tellmod = telluric.model.PCATelluricModel(pca_file, wave_min=5000)
    _tellmod.load()
    assert tellmod.wave_grid.size > _tellmod.wave_grid.size, 'Should limit wavelength range'
    assert tellmod.wave_grid[-1] == _tellmod.wave_grid[-1], 'Should only limit the blue end'

    _tellmod = telluric.model.PCATelluricModel(pca_file, wave_max=5000)
    _tellmod.load()
    assert tellmod.wave_grid.size > _tellmod.wave_grid.size, 'Should limit wavelength range'
    assert tellmod.wave_grid[0] == _tellmod.wave_grid[0], 'Should only limit the red end'

    wave_min = 9000.
    wave_max = 1.8e4
    _tellmod = telluric.model.PCATelluricModel(
        pca_file, wave_min=wave_min, wave_max=wave_max, pad_frac=0.
    )
    _tellmod.load()
    assert tellmod.wave_grid.size > _tellmod.wave_grid.size, 'Should limit wavelength range'
    assert np.all(_tellmod.wave_grid > wave_min) and np.all(_tellmod.wave_grid < wave_max), \
        'Wavelength limits should be exact because padding is 0.'

    # Test sampling the full model
    rng = np.random.default_rng(99)
    theta = rng.uniform(size=tellmod.npar)
    t = tellmod.sample_raw(theta)
    assert np.median(t) > 0.95, 'Model changed'

    # Do the same with the wavelength-limited model
    _t = _tellmod.sample_raw(theta)

    # Sample a subsection
    start = np.where(tellmod.wave_grid == _tellmod.wave_grid[0])[0][0]
    end = start+_t.size
    t = tellmod.sample_raw(theta, start=start, end=end)
    assert np.allclose(t[start:end], _t), 'Spectra should be identical'

    # Fault if the size of the parameter vector is incorrect
    with pytest.raises(PypeItError):
        t = tellmod.sample_raw(theta[:4])

    # Instantiate while setting the number of components
    npca = 4
    tellmod = telluric.model.PCATelluricModel(pca_file, npca=npca)
    tellmod.load()
    assert tellmod.npar == npca, 'Mismatch between number of PCA components requested'
    # Test getting the model
    t = tellmod.sample_raw(theta[:npca])
    # To many coefficients
    with pytest.raises(PypeItError):
        t = tellmod.sample_raw(theta)

    # Try to request more components then there are available
    npca = 15
    tellmod = telluric.model.PCATelluricModel(pca_file, npca=npca)
    tellmod.load()
    # Warning will be issued, but instantiation and load should be successful
    assert tellmod.npar < npca, 'Number of PCA components should be fewer than requested'
    t = tellmod.sample_raw(theta[:tellmod.npar])


def test_grid():

    grid_file = 'TelFit_Paranal_VIS_4900_11100_R25000.fits'

    # Test init and load
    tellmod = telluric.model.AtmGridTelluricModel(grid_file)
    tellmod.load()

    # Check some basics
    assert tellmod.wave_grid.size == 32709, 'Number of wavelengths changed'
    assert tellmod.tell_grid.shape[:-1] == (7, 9, 11, 21), 'Shape of atmospheric grid changed'

    # Test a restricted wavelength range
    _tellmod = telluric.model.AtmGridTelluricModel(grid_file, wave_min=7500)
    _tellmod.load()
    assert tellmod.wave_grid.size > _tellmod.wave_grid.size, 'Should limit wavelength range'
    assert tellmod.wave_grid[-1] == _tellmod.wave_grid[-1], 'Should only limit the blue end'

    _tellmod = telluric.model.AtmGridTelluricModel(grid_file, wave_max=7500)
    _tellmod.load()
    assert tellmod.wave_grid.size > _tellmod.wave_grid.size, 'Should limit wavelength range'
    assert tellmod.wave_grid[0] == _tellmod.wave_grid[0], 'Should only limit the red end'

    wave_min = 6000.
    wave_max = 9000.
    _tellmod = telluric.model.AtmGridTelluricModel(
        grid_file, wave_min=wave_min, wave_max=wave_max, pad_frac=0.
    )
    _tellmod.load()
    assert tellmod.wave_grid.size > _tellmod.wave_grid.size, 'Should limit wavelength range'
    assert np.all(_tellmod.wave_grid > wave_min) and np.all(_tellmod.wave_grid < wave_max), \
        'Wavelength limits should be exact because padding is 0.'

    # Test sampling the full model
    rng = np.random.default_rng(99)
    indx = np.array([
        rng.integers(a.size) for a in [
            tellmod.pressure_grid, tellmod.temp_grid, tellmod.h2o_grid, tellmod.airmass_grid
        ]
    ])
    theta = np.array([
        a[i] for i,a in zip(indx, [
            tellmod.pressure_grid, tellmod.temp_grid, tellmod.h2o_grid, tellmod.airmass_grid
        ])
    ])
    t = tellmod.sample_raw(theta)
    assert np.array_equal(t, tellmod.tell_grid[*indx]), 'Grid point selection failed'
    assert np.median(t) > 0.95, 'Model changed'

    # Do the same with the wavelength-limited model
    _t = _tellmod.sample_raw(theta)

    # Sample a subsection
    start = np.where(tellmod.wave_grid == _tellmod.wave_grid[0])[0][0]
    end = start+_t.size
    t = tellmod.sample_raw(theta, start=start, end=end)
    assert np.allclose(t[start:end], _t), 'Spectra should be identical'

    # Fault if the size of the parameter vector is incorrect
    with pytest.raises(PypeItError):
        t = tellmod.sample_raw(theta[:3])


def test_shift_stretch():

    # Build a test model
    tellmod = telluric.model.PCATelluricModel('TellPCA_3000_26000_R15000.fits')
    tellmod.load()
    rng = np.random.default_rng(99)
    theta = rng.uniform(size=tellmod.npar)
    t = tellmod.sample_raw(theta)

    _t = tellmod._shift_and_stretch(np.log10(tellmod.wave_grid), t, 0.0, 1.00)

    # TODO: I don't quite understand why this isn't a better match
    assert np.allclose(t, _t, atol=1e-6), 'Shift and stretch failure'

    _t = tellmod._shift_and_stretch(np.log10(tellmod.wave_grid), t, 10.0, 1.00)
    assert np.allclose(t[10:], _t[:-10], atol=1e-6), 'Integer shift should be better'

    # Just test basic functionality for stretch
    # TODO: Shift and stretch are not independent...
    _t = tellmod._shift_and_stretch(np.log10(tellmod.wave_grid), t, 0.0, 1.05)


def test_convolve():

    # Build a test model
    tellmod = telluric.model.PCATelluricModel('TellPCA_3000_26000_R15000.fits')
    tellmod.load()
    rng = np.random.default_rng(99)
    theta = rng.uniform(size=tellmod.npar)
    t = tellmod.sample_raw(theta)

    _t = tellmod._convolve(t, 5000)
    assert np.median(t - _t) < 1e-4, 'Median offset should be small'
    assert np.std(t - _t) < 0.1, 'Standard deviation should be smaller'


def test_sample():
    # Build a test model
    tellmod = telluric.model.PCATelluricModel('TellPCA_3000_26000_R15000.fits')
    tellmod.load()

    # Get the PCA parameters
    rng = np.random.default_rng(99)
    theta = rng.uniform(size=tellmod.npar)

    # Add the resolution, shift, and stretch
    theta = np.append(theta, [5000., 10.0, 1.02])

    # Sample the full model
    wave, tspec = tellmod.sample(theta)
    assert np.median(tspec) > 0.95, 'Transmission spectrum changed'

    # TODO: Add tests with start and end and with padding


def test_qso_pca_model():

    obj = telluric.object.QSOPCAModel('qso_pca_1200_3100.fits')

    embed()
    exit()

test_qso_pca_model()


