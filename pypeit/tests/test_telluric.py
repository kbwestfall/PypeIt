
from pathlib import Path
from IPython import embed

import numpy as np
import pytest

from pypeit import telluric
from pypeit import PypeItError
from pypeit.core import standard


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

    pcafile = 'qso_pca_1200_3100.fits'

    # Default
    obj = telluric.object.QSOPCAModel(pcafile)
    z0_redshift_wave_range = obj.wave_min, obj.wave_max
    assert obj.npca == 10, 'Wrong number of PCA components'
    assert obj.z_fid == 0., 'Wrong redshift'
    # Set the redshift
    _obj = telluric.object.QSOPCAModel(pcafile, redshift=7.0)
    assert _obj.z_fid == 7.0, 'Wrong redshift'
    assert np.isclose(_obj.wave[0] / obj.wave[0], 8.), 'Wavelength array not redshifted correctly'
    # Give an observed wavelength array
    obswave = np.linspace(20000, 24000, 512)
    obj = telluric.object.QSOPCAModel(pcafile, wave=obswave, redshift=7.0)
    assert obj.z_fid == 7.0, 'Wrong redshift'
    assert obj.wave.size == obswave.size, 'Wrong wavelength array size'
    # Limit the number of PCA components
    obj = telluric.object.QSOPCAModel(pcafile, wave=obswave, redshift=7.0, npca=7)
    assert obj.npca == 7, 'Wrong number of PCA components'
    assert obj.z_fid == 7.0, 'Wrong redshift'
    assert obj.wave.size == obswave.size, 'Wrong wavelength array size'

    # The parameter vector cannot be None for QSOPCAModel
    with pytest.raises(PypeItError):
        flux, gpm = obj.sample(None)

    rng = np.random.default_rng(99)

    # This should fail because the number of parameters is wrong.  There must be
    # at least npca parameters.
    # NOTE: This test will not work if there are too many parameters because all
    # parameters beyond the first npca are considered coefficients of the
    # polynomial.
    theta = np.append([7.0], rng.uniform(size=4))
    with pytest.raises(ValueError):
        flux, gpm = obj.sample(theta)

    # This should be successful
    theta = np.append([7.0], rng.uniform(size=obj.npca-1))
    flux, gpm = obj.sample(theta)
    # NOTE: This may fault if the number of random draws changes earlier in the
    # test; i.e., the coefficients will have changed.
    assert np.isclose(np.median(flux), 0.32486), 'QSO model changed'

    # Add a normalization
    theta = np.append(theta, [2.0])
    flux, gpm = obj.sample(theta)
    # NOTE: This divides by np.exp(2.0) because the default "model" parameter is "exp"
    assert np.isclose(np.median(flux)/np.exp(2.0), 0.32486), 'Normalization not working correctly'

    # Add a 4th order polynomial
    theta = np.append(theta, rng.uniform(size=3))
    flux, gpm = obj.sample(theta)
    assert np.isclose(np.median(flux), 2.19489), 'Normalization not working correctly'

    # Check that regions outside the wavelength range of PCA components are masked
    obswave = np.linspace(20000, 26000, 512)
    _obj = telluric.object.QSOPCAModel(pcafile, wave=obswave, redshift=7.0)
    # NOTE: This test works, but there may be a +/- 1 pixel difference for other
    # wavelength ranges...  z0_redshift_wave_range is defined at the beginning
    # of the test.
    assert np.array_equal(obswave < z0_redshift_wave_range[1]*8., _obj.spec_gpm), \
        'Pixels beyond wavelength range of PCA components should be masked'


def test_star_model():

    # Default
    ra, dec = '12:57:02.34', '+22:01:52.7'
    obj = telluric.object.StellarSpectrumModel(ra=ra, dec=dec)

    # This is exactly what is done internally for StellarSpectrumModel
    spec = standard.get_standard_spectrum(ra=ra, dec=dec)

    # Just return the spectrum
    flux, gpm = obj.sample(None)
    assert np.array_equal(spec.flux, flux), 'Stellar spectrum should be identical to standard'

    # Change the wavelength range
    obswave = np.linspace(3100, 10400, 2048)
    obj = telluric.object.StellarSpectrumModel(ra=ra, dec=dec, wave=obswave)
    flux, gpm = obj.sample(None)
    assert flux.size == obswave.size, 'Spectrum was not resampled'

    # Renormalize
    _flux, gpm = obj.sample(np.array([2.0]))
    assert np.allclose(_flux / flux, np.exp(2.)), 'Should be renormalized'

    # Change the normalization model
    obj = telluric.object.StellarSpectrumModel(ra=ra, dec=dec, wave=obswave, model='poly')
    _flux, gpm = obj.sample(np.array([2.0]))
    assert np.allclose(_flux / flux, 2.), 'Should be renormalized'

    # Normalize by a polynomial
    rng = np.random.default_rng(99)
    theta = np.append([1.1], rng.uniform(size=3))
    flux, gpm = obj.sample(theta)
    assert np.isclose(np.median(flux), 606.096), 'Median flux changed'


def test_poly_model():

    # Default wavelengths
    obj = telluric.object.PolynomialModel(model='poly')
    assert obj.wave.size == 48000, 'Default wavelength array size changed'

    # Bespoke wavelengths
    obswave = np.linspace(3100, 10400, 2048)
    obj = telluric.object.PolynomialModel(wave=obswave, model='poly')
    flux, gpm = obj.sample(None)
    assert flux.size == obswave.size, 'Spectrum shape incorrect'
    assert np.array_equal(flux, np.ones(flux.size, dtype=float)), 'Default flux should be unity.'

    # Normalize by a polynomial
    rng = np.random.default_rng(99)
    theta = np.append([1.1], rng.uniform(size=3))
    flux, gpm = obj.sample(theta)
    assert np.isclose(np.median(flux), 0.95063), 'Median flux changed'
