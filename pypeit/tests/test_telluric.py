
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

    src = telluric.source.AdjustedSpectrumModel(spec, order=0, model='poly')
    assert np.array_equal(src.wave, wave), 'Wavelength array changed'
    assert src.npar == 1, 'Number of parameters incorrect for order 0 polynomial'

    flux, gpm = src.sample(np.array([2.0]))
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
    src = telluric.source.AdjustedSpectrumModel(
        standard.PseudoStandard(wave=wave), order=5, func='legendre', model='poly'
    )

    # Test the guess parameters
    assert src.spectrum_par_guess() is None, 'Underlying spectrum should have no parameters'
    gp = src.par_guess(obs_spec)
    assert gp.size == coeffs.size, 'Bad number of parameter guesses'
    assert np.all(np.absolute(coeffs - gp) < 0.01), 'Bad parameter guess values'

    # Sample the underlying spectrum
    spec_flux, spec_gpm = src.spectrum_sample(None)
    assert np.array_equal(spec_flux, src.spec.flux), 'Should return the spectrum unmodified'
    assert np.array_equal(spec_gpm, src.spec.gpm), 'Should return the spectrum gpm unmodified'

    # Sample the model and compare to the fake spectrum
    flux_model, gpm = src.sample(gp)
    assert np.std(flux - flux_model) < 1.2*err, 'Guess model should be a better match to the data'

    # Test the parameter boundaries
    assert src.spectrum_par_bounds() is None, \
        'Underlying spectrum should have no parameter boundaries'
    rel_coeff_bounds = (-20., 20.)
    abs_coeff_bounds = (-5., 5.)
    bounds = src.par_bounds(gp, rel_coeff_bounds, abs_coeff_bounds)
    _bnds = np.asarray(bounds).T

    assert len(bounds) == coeffs.size, 'Bad number of parameter bounds'
    assert np.all(_bnds[0] < _bnds[1]), 'Lower bound is not always less than upper bound'
    assert np.all(gp > _bnds[0]) and np.all(gp < _bnds[1]), 'Guess parameters not within bounds'
    assert np.all(coeffs > _bnds[0]) and np.all(coeffs < _bnds[1]), \
        'True parameters not within bounds'
    

def test_qso_pca_model():

    pcafile = 'qso_pca_1200_3100.fits'

    # Default
    src = telluric.source.QSOPCAModel(pcafile, 0.0)
    z0_redshift_wave_range = src.wave_min, src.wave_max
    assert src.npca == 10, 'Wrong number of PCA components'
    assert src.z == 0.0, 'Wrong redshift'
    # Set the redshift
    _src = telluric.source.QSOPCAModel(pcafile, 7.0)
    assert _src.z == 7.0, 'Wrong redshift'
    assert np.isclose(_src.wave[0] / src.wave[0], 8.), 'Wavelength array not redshifted correctly'
    # Give an observed wavelength array
    obswave = np.linspace(20000, 24000, 512)
    src = telluric.source.QSOPCAModel(pcafile, 7.0, wave=obswave)
    assert src.z == 7.0, 'Wrong redshift'
    assert src.wave.size == obswave.size, 'Wrong wavelength array size'
    # Limit the number of PCA components
    src = telluric.source.QSOPCAModel(pcafile, 7.0, wave=obswave, npca=7)
    assert src.z == 7.0, 'Wrong redshift'
    assert src.npca == 7, 'Wrong number of PCA components'
    assert src.wave.size == obswave.size, 'Wrong wavelength array size'

    # The parameter vector cannot be None for QSOPCAModel
    with pytest.raises(PypeItError):
        flux, gpm = src.sample(None)

    rng = np.random.default_rng(99)

    # This should fail because the number of parameters is wrong.  There must be
    # at least npca parameters.
    # NOTE: This test will not work if there are too many parameters because all
    # parameters beyond the first npca are considered coefficients of the
    # polynomial.
    theta = np.append([7.0], rng.uniform(size=4))
    with pytest.raises(PypeItError):
        flux, gpm = src.sample(theta)

    # This should be successful
    theta = np.append([7.0], rng.uniform(size=src.npca-1))
    flux, gpm = src.sample(theta)
    # NOTE: This may fault if the number of random draws changes earlier in the
    # test; i.e., the coefficients will have changed.
    assert np.isclose(np.median(flux), 0.32486), 'QSO model changed'

    # Add a normalization
    theta = np.append(theta, [2.0])
    src = telluric.source.QSOPCAModel(pcafile, 7.0, wave=obswave, npca=7, order=0)
    flux, gpm = src.sample(theta)
    # NOTE: This divides by np.exp(2.0) because the default "model" parameter is "exp"
    assert np.isclose(np.median(flux)/np.exp(2.0), 0.32486), 'Normalization not working correctly'

    # Add a 4th order polynomial
    theta = np.append(theta, rng.uniform(size=3))
    src = telluric.source.QSOPCAModel(pcafile, 7.0, wave=obswave, npca=7, order=3)
    flux, gpm = src.sample(theta)
    assert np.isclose(np.median(flux), 2.19489), 'Normalization not working correctly'

    # Check that regions outside the wavelength range of PCA components are masked
    obswave = np.linspace(20000, 26000, 512)
    _src = telluric.source.QSOPCAModel(pcafile, 7.0, wave=obswave)
    # NOTE: This test works, but there may be a +/- 1 pixel difference for other
    # wavelength ranges...  z0_redshift_wave_range is defined at the beginning
    # of the test.
    assert np.array_equal(obswave < z0_redshift_wave_range[1]*8., _src.spec_gpm), \
        'Pixels beyond wavelength range of PCA components should be masked'


def test_qso_pca_model_sim():
    pcafile = 'qso_pca_1200_3100.fits'
    obswave = np.linspace(20000, 24000, 512)
    order = 3
    src = telluric.source.QSOPCAModel(
        pcafile, 7.0, wave=obswave, order=order, func='legendre', model='poly'
    )

    # Create a synthetic spectrum
    rng = np.random.default_rng(99)
    spec_theta = np.append([6.98], rng.uniform(size=src.npca-1))

    spec_flux, spec_gpm = src.spectrum_sample(spec_theta)
    with pytest.raises(PypeItError):
        # Should fail because the number of parameters is wrong
        f, g = src.sample(spec_theta)

    # Add the polynomial coefficients
    theta = np.concatenate((spec_theta, [3.0], rng.uniform(size=order)))
    flux, gpm = src.sample(theta)

    err = 0.03
    flux += rng.normal(scale=err, size=flux.size)
    obs_spec = spectrum.Spectrum(
        wave=obswave, flux=flux, ivar=np.full(flux.size, 1/err**2), gpm=gpm
    )

    gp = src.par_guess(obs_spec)
    assert gp.size == src.npar, 'Bad number of parameter guesses'
    # Test the polynomial coefficients
    # NOTE: The guess redshift (7.0) is different enough from the known value
    # (6.98) that the polynomial coefficients are affected.
    assert np.all(gp[-order-1:] - theta[-order-1:] < 0.1), 'Bad parameter guess values'

    # Sample the model and compare to the fake spectrum
    flux_model, gpm_model = src.sample(gp)
    # The difference is dominated by the difference in redshift
    assert np.std((obs_spec.flux - flux_model)[obs_spec.gpm & gpm_model]) < 3 * err, \
        'Guess model should be a better match to the data'

    rel_coeff_bounds = (-20., 20.)
    abs_coeff_bounds = (-5., 5.)
    bounds = src.par_bounds(gp, rel_coeff_bounds, abs_coeff_bounds)
    _bnds = np.asarray(bounds).T
    assert len(bounds) == theta.size, 'Bad number of parameter bounds'
    assert np.all(_bnds[0] < _bnds[1]), 'Lower bound is not always less than upper bound'
    assert np.all(gp > _bnds[0]) and np.all(gp < _bnds[1]), 'Guess parameters not within bounds'
    assert np.all(theta > _bnds[0]) and np.all(theta < _bnds[1]), \
        'True parameters not within bounds'

def test_star_model():

    # Default
    ra, dec = '12:57:02.34', '+22:01:52.7'
    src = telluric.source.StellarSpectrumModel(ra=ra, dec=dec)

    # This is exactly what is done internally for StellarSpectrumModel
    spec = standard.get_standard_spectrum(ra=ra, dec=dec)

    # Just return the spectrum
    flux, gpm = src.sample(None)
    assert np.array_equal(spec.flux, flux), 'Stellar spectrum should be identical to standard'

    # Change the wavelength range
    obswave = np.linspace(3100, 10400, 2048)
    src = telluric.source.StellarSpectrumModel(ra=ra, dec=dec, wave=obswave)
    flux, gpm = src.sample(None)
    assert flux.size == obswave.size, 'Spectrum was not resampled'

    # Renormalize
    src = telluric.source.StellarSpectrumModel(ra=ra, dec=dec, wave=obswave, order=0)
    _flux, gpm = src.sample(np.array([2.0]))
    assert np.allclose(_flux / flux, np.exp(2.)), 'Should be renormalized'

    # Change the normalization model
    src = telluric.source.StellarSpectrumModel(ra=ra, dec=dec, wave=obswave, order=0, model='poly')
    _flux, gpm = src.sample(np.array([2.0]))
    assert np.allclose(_flux / flux, 2.), 'Should be renormalized'

    # Normalize by a polynomial
    rng = np.random.default_rng(99)
    theta = np.append([1.1], rng.uniform(size=3))
    src = telluric.source.StellarSpectrumModel(ra=ra, dec=dec, wave=obswave, order=3, model='poly')
    flux, gpm = src.sample(theta)
    assert np.isclose(np.median(flux), 606.096), 'Median flux changed'


def test_poly_model():

    # Default wavelengths
    src = telluric.source.PolynomialModel(model='poly')
    assert src.wave.size == 48000, 'Default wavelength array size changed'

    # Bespoke wavelengths
    obswave = np.linspace(3100, 10400, 2048)
    src = telluric.source.PolynomialModel(wave=obswave, model='poly')
    flux, gpm = src.sample(None)
    assert flux.size == obswave.size, 'Spectrum shape incorrect'
    assert np.array_equal(flux, np.ones(flux.size, dtype=float)), 'Default flux should be unity.'

    # Normalize by a polynomial
    rng = np.random.default_rng(99)
    theta = np.append([1.1], rng.uniform(size=3))
    src = telluric.source.PolynomialModel(wave=obswave, order=3, model='poly')
    flux, gpm = src.sample(theta)
    assert np.isclose(np.median(flux), 0.95063), 'Median flux changed'


def test_fitter():

    # Random number generator
    rng = np.random.default_rng(99)

    # Create the wavelength array
    wave = np.linspace(9000, 18000, 4000)

    # Create a telluric model
    tellmod = telluric.model.PCATelluricModel(
        'TellPCA_3000_26000_R15000.fits', wave_min=wave[0]/1.1, wave_max=wave[-1]*1.1
    )
    # Generate some test parameters
    tellmod_par = np.append(rng.uniform(size=tellmod.base_npar), [2000., 0.0, 1.00])

    # Create a simple source model with the right order and function.
    # NOTE: Importantly, this uses the same wavelength vector as the telluric
    # model.
    order = 5
    src = telluric.source.AdjustedSpectrumModel(
        standard.PseudoStandard(wave=tellmod.wave), order=order, func='legendre', model='poly'
    )
    # Generate some test parameters
    src_par = np.append([10.], rng.uniform(size=order))

    # Instantiate the fitter
    fitter = telluric.fitter.ObservedSourceModel(src, tellmod)
    # And set the true model parameters
    tp = np.concatenate((src_par, tellmod_par))

    # Use the model to generate a fake spectrum
    src_flux, src_gpm = src.sample(src_par)
    tell_wave, tell_flux, tell_gpm = tellmod.sample(tellmod_par)
    fit_wave, fit_flux, fit_gpm = fitter.sample(tp)

    # Test the construction of the observed model spectrum
    assert np.array_equal(fit_flux, src_flux*tell_flux), 'Observed spectrum sampling failed'

    # Create a synthetic spectrum and resample it to the original wavelength array
    obs_spec = spectrum.Spectrum(
        wave=fit_wave, flux=fit_flux, ivar=np.ones(fit_flux.size), gpm=fit_gpm
    ).resample(wave)
    # Add noise and re-init
    err = 0.1
    obs_spec = spectrum.Spectrum(
        wave=wave, flux=obs_spec.flux + rng.normal(scale=err, size=wave.size),
        ivar=np.full(wave.size, 1/err**2), gpm=obs_spec.gpm
    )

    # Get the parameter guesses and bounds
    gp = fitter.par_guess(obs_spec)
    # NOTE: A difference of 1 is arbitrary here, but it works in practice
    assert np.all(np.absolute(gp - tp) < 1), 'Parameter guesses should be better'
    bp = fitter.par_bounds(gp)

    # Test the figure-of-merit at the guess parameters
    with pytest.raises(PypeItError):
        # Should fail because the observed spectrum hasn't been set
        fom = fitter.fit_fom(gp)

    fitter.obs_spec = obs_spec
    with pytest.raises(PypeItError):
        # Should fail because the observed spectrum doesn't have the same
        # wavelength array as the model
        fom = fitter.fit_fom(gp)

    fitter.obs_spec = obs_spec.resample(fitter.wave)
    fom = fitter.fit_fom(gp)
    assert fom / np.sum(fitter.obs_spec.gpm) > 0.9, 'Fit FOM at guess parameters changed'

    popsize = 30
    ballsize = 5e-4

    # All guess parameters are set
    init = fitter._init_fit_pop(bp, gp, popsize, ballsize, 99)
    lb, ub = np.asarray(bp).T
    assert init.shape == (popsize * fitter.npar, fitter.npar), 'Initial population shape incorrect'
    assert np.all(init >= lb) and np.all(init <= ub), 'Initial population out of bounds'

    # Set all the telluric model parameters to None so that they're defined
    # using the latin hypercube
    _gp = np.asarray(gp, dtype=object)
    _gp[src.npar:] = None

    _init = fitter._init_fit_pop(bp, _gp, popsize, ballsize, 99) 
    assert _init.shape == (popsize * fitter.npar, fitter.npar), 'Initial population shape incorrect'
    assert np.all(_init >= lb) and np.all(_init <= ub), 'Initial population out of bounds'

#    # TODO: This takes a long time to complete but it should be successful.  Add
#    # it to the dev-suite unit tests!
#    best_fit_par = fitter.fit(obs_spec, bp)
#    bf_wave, bf_flux, bf_gpm = fitter.sample(best_fit_par)
#    bf_spec = spectrum.Spectrum(wave=bf_wave, flux=bf_flux, gpm=bf_gpm).resample(obs_spec.wave)
#    assert np.std((obs_spec.flux - bf_spec.flux)[obs_spec.gpm & bf_spec.gpm]) < 1.2 * err, \
#        'Best-fit model should be a better match to the data'

    # Test the fitting when starting near the guess parameters
    best_fit_par = fitter.fit(
        obs_spec, bp, guess_par=gp, popsize=popsize, ballsize=ballsize, rng=rng
    )
    # NOTE: A difference of 1 is arbitrary here, but it works in practice.  The
    # median is used basically so it ignores the difference in the resolution,
    # which show a small relative differnce but a large absolute difference.
    assert np.median(np.absolute(best_fit_par - tp)) < 1, \
        'Best-fit parameters should be closer to the true parameters'
    bf_wave, bf_flux, bf_gpm = fitter.sample(best_fit_par)
    bf_spec = spectrum.Spectrum(wave=bf_wave, flux=bf_flux, gpm=bf_gpm).resample(obs_spec.wave)
    assert np.std((obs_spec.flux - bf_spec.flux)[obs_spec.gpm & bf_spec.gpm]) < 1.2 * err, \
        'Best-fit model should be a better match to the data'
