
from pathlib import Path
from IPython import embed

import numpy as np
import pytest

from pypeit import telluric
from pypeit import PypeItError
from pypeit.core import coadd
from pypeit.core import spectrum
from pypeit.core import standard
from pypeit.core.wavecal import wvutils


def test_init_model():
    test_file = Path('test.fits')
    if test_file.is_file():
        test_file.unlink()

    # File does not exist
    with pytest.raises(PypeItError):
        tellmod = telluric.model.PCATelluricModel(test_file)

    # File exists ...
    test_file.touch()
    # ... but load function undefined
    with pytest.raises(PypeItError):
        tellmod = telluric.model.TelluricModel(test_file)
    # Should not fail if not loaded
    tellmod = telluric.model.TelluricModel(test_file, load=False)
    # ... but will fail when trying to load
    with pytest.raises(PypeItError):
        tellmod.load()
    # Remove the empty file
    test_file.unlink()

    tellmod = telluric.model.PCATelluricModel('TellPCA_3000_26000_R15000.fits')


def test_pca_model():

    pca_file = 'TellPCA_3000_26000_R15000.fits'

    # Test init and load
    tellmod = telluric.model.PCATelluricModel(pca_file)

    # Check some basics
    assert tellmod.wave.size == 82398, 'Number of wavelengths changed'
    assert tellmod.tell_grid.shape[0] == 11, 'Number of components changed'

    # Test a restricted wavelength range
    _tellmod = telluric.model.PCATelluricModel(pca_file, wave_min=5000)
    assert tellmod.wave.size > _tellmod.wave.size, 'Should limit wavelength range'
    assert tellmod.wave[-1] == _tellmod.wave[-1], 'Should only limit the blue end'

    _tellmod = telluric.model.PCATelluricModel(pca_file, wave_max=5000)
    assert tellmod.wave.size > _tellmod.wave.size, 'Should limit wavelength range'
    assert tellmod.wave[0] == _tellmod.wave[0], 'Should only limit the red end'

    wave_min = 9000.
    wave_max = 1.8e4
    _tellmod = telluric.model.PCATelluricModel(pca_file, wave_min=wave_min, wave_max=wave_max)
    assert tellmod.wave.size > _tellmod.wave.size, 'Should limit wavelength range'
    assert np.all(_tellmod.wave > wave_min) and np.all(_tellmod.wave < wave_max), \
        'Wavelength limits should be exact because padding is 0.'

    # Test sampling the full model
    rng = np.random.default_rng(99)
    theta = rng.uniform(size=tellmod.base_npar)
    t = tellmod.base_sample(theta)[1]
    assert np.median(t) > 0.95, 'Model changed'

    # Do the same with the wavelength-limited model
    _w, _t, _gpm = _tellmod.base_sample(theta)

    # Sample a subsection
    tellmod.restrict_wave_range(wave_min=wave_min, wave_max=wave_max, pad_frac=0.0)
    assert (tellmod.wave[tellmod.s_wave] > wave_min 
        and tellmod.wave[tellmod.s_wave-1] < wave_min
    ), 'Minimum wavelength restriction failed'
    assert (tellmod.wave[tellmod.e_wave-1] < wave_max
        and tellmod.wave[tellmod.e_wave] > wave_max
    ), 'Maximum wavelength restriction failed'
    w, t, gpm = tellmod.base_sample(theta)
    assert np.allclose(t[gpm], _t), 'Spectra should be identical'
    tellmod.restrict_wave_range()

    # Fault if the size of the parameter vector is incorrect
    with pytest.raises(PypeItError):
        t = tellmod.base_sample(theta[:4])[1]

    # Instantiate while setting the number of components
    npca = 4
    tellmod = telluric.model.PCATelluricModel(pca_file, npca=npca)
    assert tellmod.base_npar == npca-1, 'Mismatch between number of model parameters and the number of PCA components requested'
    # Test getting the model
    t = tellmod.base_sample(theta[:npca-1])[1]
    # To many coefficients
    with pytest.raises(PypeItError):
        t = tellmod.base_sample(theta)

    # Should fail if the number of components is less than 2
    with pytest.raises(PypeItError):
        tellmod = telluric.model.PCATelluricModel(pca_file, npca=1)

    # Try to request more components than are available
    npca = 15
    tellmod = telluric.model.PCATelluricModel(pca_file, npca=npca)
    # Warning will be issued, but instantiation and load should be successful
    assert tellmod.base_npar < npca-1, 'Number of PCA components should be fewer than requested'
    t = tellmod.base_sample(theta[:tellmod.base_npar])


def test_grid_model():

    grid_file = 'TelFit_Paranal_VIS_4900_11100_R25000.fits'

    # Test init and load
    tellmod = telluric.model.AtmGridTelluricModel(grid_file)

    # Check some basics
    assert tellmod.wave.size == 32709, 'Number of wavelengths changed'
    assert tellmod.tell_grid.shape[:-1] == (7, 9, 11, 21), 'Shape of atmospheric grid changed'

    # Test a restricted wavelength range
    _tellmod = telluric.model.AtmGridTelluricModel(grid_file, wave_min=7500)
    assert tellmod.wave.size > _tellmod.wave.size, 'Should limit wavelength range'
    assert tellmod.wave[-1] == _tellmod.wave[-1], 'Should only limit the blue end'

    _tellmod = telluric.model.AtmGridTelluricModel(grid_file, wave_max=7500)
    assert tellmod.wave.size > _tellmod.wave.size, 'Should limit wavelength range'
    assert tellmod.wave[0] == _tellmod.wave[0], 'Should only limit the red end'

    wave_min = 6000.
    wave_max = 9000.
    _tellmod = telluric.model.AtmGridTelluricModel(grid_file, wave_min=wave_min, wave_max=wave_max)
    assert tellmod.wave.size > _tellmod.wave.size, 'Should limit wavelength range'
    assert np.all(_tellmod.wave > wave_min) and np.all(_tellmod.wave < wave_max), \
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
    t = tellmod.base_sample(theta)[1]
    assert np.array_equal(t, tellmod.tell_grid[*indx]), 'Grid point selection failed'
    assert np.median(t) > 0.95, 'Model changed'

    # Do the same with the wavelength-limited model
    _w, _t, _gpm = _tellmod.base_sample(theta)

    # Sample a subsection
    tellmod.restrict_wave_range(wave_min=wave_min, wave_max=wave_max, pad_frac=0.0)
    assert (tellmod.wave[tellmod.s_wave] > wave_min
        and tellmod.wave[tellmod.s_wave-1] < wave_min
    ), 'Minimum wavelength restriction failed'
    assert (tellmod.wave[tellmod.e_wave-1] < wave_max
        and tellmod.wave[tellmod.e_wave] > wave_max
    ), 'Maximum wavelength restriction failed'
    w, t, gpm = tellmod.base_sample(theta)
    assert np.allclose(t[gpm], _t), 'Spectra should be identical'
    tellmod.restrict_wave_range()

    # Fault if the size of the parameter vector is incorrect
    with pytest.raises(PypeItError):
        t = tellmod.base_sample(theta[:3])


def test_shift_stretch():

    # Build a test model
    tellmod = telluric.model.PCATelluricModel('TellPCA_3000_26000_R15000.fits')
    rng = np.random.default_rng(99)
    theta = rng.uniform(size=tellmod.base_npar)
    t = tellmod.base_sample(theta)[1]

    _t = tellmod._shift_and_stretch(np.log10(tellmod.wave), t, 0.0, 1.00)

    # TODO: I don't understand why this isn't a better match
    assert np.allclose(t, _t, atol=1e-6), 'Shift and stretch failure'

    _t = tellmod._shift_and_stretch(np.log10(tellmod.wave), t, 10.0, 1.00)
    assert np.allclose(t[10:], _t[:-10], atol=1e-6), 'Integer shift should be better'

    # Just test basic functionality for stretch
    # TODO: Shift and stretch are not independent...
    _t = tellmod._shift_and_stretch(np.log10(tellmod.wave), t, 0.0, 1.05)


def test_convolve():

    # Build a test model
    tellmod = telluric.model.PCATelluricModel('TellPCA_3000_26000_R15000.fits')
    rng = np.random.default_rng(99)
    theta = rng.uniform(size=tellmod.base_npar)
    t = tellmod.base_sample(theta)[1]

    _t = tellmod._convolve(t, 5000)
    assert np.median(t - _t) < 1e-4, 'Median offset should be small'
    assert np.std(t - _t) < 0.1, 'Standard deviation should be smaller'


def test_sample_model():
    # Build a test model
    tellmod = telluric.model.PCATelluricModel('TellPCA_3000_26000_R15000.fits')

    # Get the PCA parameters
    rng = np.random.default_rng(99)
    theta = rng.uniform(size=tellmod.base_npar)

    # Add the resolution, shift, and stretch
    theta = np.append(theta, [5000., 10.0, 1.02])

    # Sample the full model
    wave, tspec, gpm = tellmod.sample(theta)
    assert np.median(tspec) > 0.95, 'Transmission spectrum changed'

    # TODO: Add tests with start and end and with padding


def test_adj_spectrum_model():
    wave = np.linspace(3000, 10000, 1000)
    spec = standard.PseudoStandard(wave=wave)

    obj = telluric.object.AdjustedSpectrumModel(spec, order=0, model='poly')
    assert np.array_equal(obj.wave, wave), 'Wavelength array changed'
    assert obj.npar == 1, 'Number of parameters incorrect for order 0 polynomial'

    flux, gpm = obj.sample(np.array([2.0]))
    assert np.allclose(flux, spec.flux * 2.0), 'Sampled model is wrong'
    assert np.all(gpm), 'Good pixel mask is wrong'


def test_adj_spectrum_model_sim():

    rng = np.random.default_rng(99)

    # Make a fake spectrum, and make sure it is always positive
    wave = np.linspace(3000, 10000, 1000)
    coeffs = np.append([10.], rng.uniform(size=5))
    flux = coadd.poly_model_eval(coeffs, 'legendre', 'poly', wave, wave[0], wave[-1])
    err = 0.1
    flux += rng.normal(scale=err, size=flux.size)
    obs_spec = spectrum.Spectrum(
        wave=wave, flux=flux, ivar=np.full(flux.size, 1/err**2), gpm=np.ones(flux.size, dtype=bool)
    )

    # Create a simple model with the right order and function
    obj = telluric.object.AdjustedSpectrumModel(
        standard.PseudoStandard(wave=wave), order=5, func='legendre', model='poly'
    )

    # Test the guess parameters
    assert obj.spectrum_par_guess() is None, 'Underlying spectrum should have no parameters'
    gp = obj.par_guess(obs_spec)
    assert gp.size == coeffs.size, 'Bad number of parameter guesses'
    assert np.all(np.absolute(coeffs - gp) < 0.01), 'Bad parameter guess values'

    # Sample the underlying spectrum
    spec_flux, spec_gpm = obj.spectrum_sample(None)
    assert np.array_equal(spec_flux, obj.spec.flux), 'Should return the spectrum unmodified'
    assert np.array_equal(spec_gpm, obj.spec.gpm), 'Should return the spectrum gpm unmodified'

    # Sample the model and compare to the fake spectrum
    flux_model, gpm = obj.sample(gp)
    assert np.std(flux - flux_model) < 1.2*err, 'Guess model should be a better match to the data'

    # Test the parameter boundaries
    assert obj.spectrum_par_bounds() is None, \
        'Underlying spectrum should have no parameter boundaries'
    rel_coeff_bounds = (-20., 20.)
    abs_coeff_bounds = (-5., 5.)
    bounds = obj.par_bounds(gp, rel_coeff_bounds, abs_coeff_bounds)
    _bnds = np.asarray(bounds).T

    assert len(bounds) == coeffs.size, 'Bad number of parameter bounds'
    assert np.all(_bnds[0] < _bnds[1]), 'Lower bound is not always less than upper bound'
    assert np.all(gp > _bnds[0]) and np.all(gp < _bnds[1]), 'Guess parameters not within bounds'
    assert np.all(coeffs > _bnds[0]) and np.all(coeffs < _bnds[1]), \
        'True parameters not within bounds'
    

def test_qso_pca_model():

    pcafile = 'qso_pca_1200_3100.fits'

    # Default
    obj = telluric.object.QSOPCAModel(pcafile, 0.0)
    z0_redshift_wave_range = obj.wave_min, obj.wave_max
    assert obj.npca == 10, 'Wrong number of PCA components'
    assert obj.z == 0.0, 'Wrong redshift'
    # Set the redshift
    _obj = telluric.object.QSOPCAModel(pcafile, 7.0)
    assert _obj.z == 7.0, 'Wrong redshift'
    assert np.isclose(_obj.wave[0] / obj.wave[0], 8.), 'Wavelength array not redshifted correctly'
    # Give an observed wavelength array
    obswave = np.linspace(20000, 24000, 512)
    obj = telluric.object.QSOPCAModel(pcafile, 7.0, wave=obswave)
    assert obj.z == 7.0, 'Wrong redshift'
    assert obj.wave.size == obswave.size, 'Wrong wavelength array size'
    # Limit the number of PCA components
    obj = telluric.object.QSOPCAModel(pcafile, 7.0, wave=obswave, npca=7)
    assert obj.z == 7.0, 'Wrong redshift'
    assert obj.npca == 7, 'Wrong number of PCA components'
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
    with pytest.raises(PypeItError):
        flux, gpm = obj.sample(theta)

    # This should be successful
    theta = np.append([7.0], rng.uniform(size=obj.npca-1))
    flux, gpm = obj.sample(theta)
    # NOTE: This may fault if the number of random draws changes earlier in the
    # test; i.e., the coefficients will have changed.
    assert np.isclose(np.median(flux), 0.32486), 'QSO model changed'

    # Add a normalization
    theta = np.append(theta, [2.0])
    obj = telluric.object.QSOPCAModel(pcafile, 7.0, wave=obswave, npca=7, order=0)
    flux, gpm = obj.sample(theta)
    # NOTE: This divides by np.exp(2.0) because the default "model" parameter is "exp"
    assert np.isclose(np.median(flux)/np.exp(2.0), 0.32486), 'Normalization not working correctly'

    # Add a 4th order polynomial
    theta = np.append(theta, rng.uniform(size=3))
    obj = telluric.object.QSOPCAModel(pcafile, 7.0, wave=obswave, npca=7, order=3)
    flux, gpm = obj.sample(theta)
    assert np.isclose(np.median(flux), 2.19489), 'Normalization not working correctly'

    # Check that regions outside the wavelength range of PCA components are masked
    obswave = np.linspace(20000, 26000, 512)
    _obj = telluric.object.QSOPCAModel(pcafile, 7.0, wave=obswave)
    # NOTE: This test works, but there may be a +/- 1 pixel difference for other
    # wavelength ranges...  z0_redshift_wave_range is defined at the beginning
    # of the test.
    assert np.array_equal(obswave < z0_redshift_wave_range[1]*8., _obj.spec_gpm), \
        'Pixels beyond wavelength range of PCA components should be masked'


def test_qso_pca_model_sim():
    pcafile = 'qso_pca_1200_3100.fits'
    obswave = np.linspace(20000, 24000, 512)
    order = 3
    obj = telluric.object.QSOPCAModel(
        pcafile, 7.0, wave=obswave, order=order, func='legendre', model='poly'
    )

    # Create a synthetic spectrum
    rng = np.random.default_rng(99)
    spec_theta = np.append([6.98], rng.uniform(size=obj.npca-1))

    spec_flux, spec_gpm = obj.spectrum_sample(spec_theta)
    with pytest.raises(PypeItError):
        # Should fail because the number of parameters is wrong
        f, g = obj.sample(spec_theta)

    # Add the polynomial coefficients
    theta = np.concatenate((spec_theta, [3.0], rng.uniform(size=order)))
    flux, gpm = obj.sample(theta)

    err = 0.03
    flux += rng.normal(scale=err, size=flux.size)
    obs_spec = spectrum.Spectrum(
        wave=obswave, flux=flux, ivar=np.full(flux.size, 1/err**2), gpm=gpm
    )

    gp = obj.par_guess(obs_spec)
    assert gp.size == obj.npar, 'Bad number of parameter guesses'
    # Test the polynomial coefficients
    # NOTE: The guess redshift (7.0) is different enough from the known value
    # (6.98) that the polynomial coefficients are affected.
    assert np.all(gp[-order-1:] - theta[-order-1:] < 0.1), 'Bad parameter guess values'

    # Sample the model and compare to the fake spectrum
    flux_model, gpm_model = obj.sample(gp)
    # The difference is dominated by the difference in redshift
    assert np.std((obs_spec.flux - flux_model)[obs_spec.gpm & gpm_model]) < 3 * err, \
        'Guess model should be a better match to the data'

    rel_coeff_bounds = (-20., 20.)
    abs_coeff_bounds = (-5., 5.)
    bounds = obj.par_bounds(gp, rel_coeff_bounds, abs_coeff_bounds)
    _bnds = np.asarray(bounds).T
    assert len(bounds) == theta.size, 'Bad number of parameter bounds'
    assert np.all(_bnds[0] < _bnds[1]), 'Lower bound is not always less than upper bound'
    assert np.all(gp > _bnds[0]) and np.all(gp < _bnds[1]), 'Guess parameters not within bounds'
    assert np.all(theta > _bnds[0]) and np.all(theta < _bnds[1]), \
        'True parameters not within bounds'

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
    obj = telluric.object.StellarSpectrumModel(ra=ra, dec=dec, wave=obswave, order=0)
    _flux, gpm = obj.sample(np.array([2.0]))
    assert np.allclose(_flux / flux, np.exp(2.)), 'Should be renormalized'

    # Change the normalization model
    obj = telluric.object.StellarSpectrumModel(ra=ra, dec=dec, wave=obswave, order=0, model='poly')
    _flux, gpm = obj.sample(np.array([2.0]))
    assert np.allclose(_flux / flux, 2.), 'Should be renormalized'

    # Normalize by a polynomial
    rng = np.random.default_rng(99)
    theta = np.append([1.1], rng.uniform(size=3))
    obj = telluric.object.StellarSpectrumModel(ra=ra, dec=dec, wave=obswave, order=3, model='poly')
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
    obj = telluric.object.PolynomialModel(wave=obswave, order=3, model='poly')
    flux, gpm = obj.sample(theta)
    assert np.isclose(np.median(flux), 0.95063), 'Median flux changed'


def test_fitter():

    # Random number generator
    rng = np.random.default_rng(99)

    # Make a fake spectrum
    wave = np.linspace(7000, 18000, 10000)
    resolution = wvutils.get_sampling(wave)[2]
    order = 5
    func = 'legendre'
    model = 'poly'
    poly_coeffs = np.append([10.], rng.uniform(size=order))
    flux = coadd.poly_model_eval(poly_coeffs, func, model, wave, wave[0], wave[-1])

    # Create a telluric model
    tellmod = telluric.model.PCATelluricModel(
        'TellPCA_3000_26000_R15000.fits', wave_min=0.9*wave[0], wave_max=1.1*wave[-1]
    )
    # Generate some test parameters
    tellmod_par = np.append(rng.uniform(size=tellmod.base_npar), [2000., 0.0, 1.00])

    # Add the telluric absorption to the fake spectrum
    twave, tspec, tgpm = tellmod.sample(tellmod_par)
    tell_spec = spectrum.Spectrum(wave=twave, flux=tspec, gpm=tgpm).resample(wave)
    flux *= tell_spec.flux

    # Add noise
    err = 0.1
    flux += rng.normal(scale=err, size=flux.size)

    # Construct the synthetic spectrum object
    obs_spec = spectrum.Spectrum(
        wave=wave, flux=flux, ivar=np.full(flux.size, 1/err**2), gpm=np.ones(flux.size, dtype=bool)
    )

    # Create a simple model with the right order and function.
    # NOTE: Importantly, this uses the same wavelength vector as the telluric
    # model.
    obj = telluric.object.AdjustedSpectrumModel(
        standard.PseudoStandard(wave=tellmod.wave), order=order, func=func, model=model
    )

    # Instantiate the fitter
    fitter = telluric.fitter.ObservedSourceModel(obj, tellmod)

    # Get the parameter guesses
    gp = fitter.par_guess(obs_spec)
    bp = fitter.par_bounds(gp)

    embed()
    exit()

test_fitter()