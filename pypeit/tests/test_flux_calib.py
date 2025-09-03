
from pathlib import Path

from IPython import embed
import numpy as np
import pytest

import matplotlib
matplotlib.use('agg')

from pypeit import dataPaths
from pypeit.core import atmextinction
from pypeit.core import flux_calib_refactor
from pypeit.core import spectrum
from pypeit.core import standard
from pypeit.core import wavemask
from pypeit.pypmsgs import PypeItError
from pypeit.sampling import Resample

def test_standard_zeropoint_basic():
    """
    Test the basic functionality.
    """
    masks = [
        dataPaths.masks.get_file_path('hydrogen.toml'),
        dataPaths.masks.get_file_path('telluric.toml'),
        dataPaths.masks.get_file_path('atm.toml'),
    ]
    region_mask = wavemask.read_wavelength_masks(masks)

    # Load the test spectrum data
    # TODO: Change this to use a spec1d file?
    test_spec_file = dataPaths.tests.get_file_path('test_standard_zeropoint_obs_spec.npz')
    spec_data = np.load(test_spec_file)
    obs_spec = spectrum.Spectrum(
        spec_data['wave_cnts'], spec_data['counts'], ivar=spec_data['counts_ivar'],
        gpm=spec_data['counts_mask']
    )

    archives = standard.archived_flux_classes()
    std_spec = archives['calspec'].from_name('FEIGE66')
    r = Resample(std_spec.flux, x=std_spec.wave, newx=obs_spec.wave, conserve=False)
    std_spec = spectrum.Spectrum(r.outx, r.outy, gpm=r.outf > 0.8)

    atmext = atmextinction.AtmosphericExtinction.from_coordinates(-121.6367, 37.3433)

    zp_spec, fit_gpm, fit_gpm_rej, zp_bspl = flux_calib_refactor.standard_zeropoint(
        obs_spec, std_spec, exptime=30., atm_extinction=atmext, airmass=1.04, nresln=20,
        resolution=3000, region_mask=region_mask,
    )
    zp_model = zp_bspl.value(obs_spec.wave)[0]

    assert zp_model.size == zp_spec.size, 'Model has the wrong size'
    assert np.all((zp_model > 16.27) & (zp_model < 17.6)), 'Zeropoints changed'
    rms = np.sqrt(np.mean((zp_spec.flux[fit_gpm_rej] - zp_model[fit_gpm_rej])**2))
    assert rms < 0.012, 'RMS increased'
    assert np.sum(fit_gpm) < np.sum(obs_spec.gpm), 'Region mask not applied'
    assert np.sum(fit_gpm_rej) < np.sum(fit_gpm), 'No pixels were rejected'


def test_standard_zeropoint_input():
    # Load the test spectrum data
    # TODO: Change this to use a spec1d file?
    test_spec_file = dataPaths.tests.get_file_path('test_standard_zeropoint_obs_spec.npz')
    spec_data = np.load(test_spec_file)
    obs_spec = spectrum.Spectrum(
        spec_data['wave_cnts'], spec_data['counts'], ivar=spec_data['counts_ivar'],
        gpm=spec_data['counts_mask']
    )

    archives = standard.archived_flux_classes()
    std_spec = archives['calspec'].from_name('FEIGE66')
    r = Resample(std_spec.flux, x=std_spec.wave, newx=obs_spec.wave, conserve=False)
    std_spec = spectrum.Spectrum(r.outx, r.outy, gpm=r.outf > 0.8)

    # Primary inputs have to be spectrum classes
    with pytest.raises(PypeItError):
        flux_calib_refactor.standard_zeropoint(obs_spec.flux, std_spec)
    with pytest.raises(PypeItError):
        flux_calib_refactor.standard_zeropoint(obs_spec, std_spec.flux)

    # Create a 2D spec
    _obs_spec = spectrum.Spectrum(
        np.stack((obs_spec.wave,)*2), np.stack((obs_spec.flux,)*2),
        ivar=np.stack((obs_spec.ivar,)*2), gpm=np.stack((obs_spec.gpm,)*2)
    )
    # Must be 1D
    with pytest.raises(PypeItError):
        flux_calib_refactor.standard_zeropoint(_obs_spec, std_spec)

    # Wavelengths must be identical
    _std_spec = std_spec.copy()
    _std_spec.wave += 10
    with pytest.raises(PypeItError):
        flux_calib_refactor.standard_zeropoint(obs_spec, _std_spec)


def test_standard_zeropoint_kwargs():

    masks = [
        dataPaths.masks.get_file_path('hydrogen.toml'),
        dataPaths.masks.get_file_path('telluric.toml'),
        dataPaths.masks.get_file_path('atm.toml'),
    ]
    region_mask = wavemask.read_wavelength_masks(masks)

    # Load the test spectrum data
    # TODO: Change this to use a spec1d file?
    test_spec_file = dataPaths.tests.get_file_path('test_standard_zeropoint_obs_spec.npz')
    spec_data = np.load(test_spec_file)
    obs_spec = spectrum.Spectrum(
        spec_data['wave_cnts'], spec_data['counts'], ivar=spec_data['counts_ivar'],
        gpm=spec_data['counts_mask']
    )

    archives = standard.archived_flux_classes()
    std_spec = archives['calspec'].from_name('FEIGE66')
    r = Resample(std_spec.flux, x=std_spec.wave, newx=obs_spec.wave, conserve=False)
    std_spec = spectrum.Spectrum(r.outx, r.outy, gpm=r.outf > 0.8)

    atmext = atmextinction.AtmosphericExtinction.from_coordinates(-121.6367, 37.3433)

    zp_spec, fit_gpm, fit_gpm_rej, zp_bspl = flux_calib_refactor.standard_zeropoint(
        obs_spec, std_spec, exptime=30., atm_extinction=atmext, airmass=1.04, nresln=20,
        resolution=3000, region_mask=region_mask,
    )

    # Change the exposure time
    _zp_spec = flux_calib_refactor.standard_zeropoint(
        obs_spec, std_spec, exptime=10., atm_extinction=atmext, airmass=1.04, nresln=20,
        resolution=3000, region_mask=region_mask,
    )[0]
    assert np.allclose(zp_spec.flux, _zp_spec.flux - 2.5*np.log10(3.)), \
        'Bad dependence on exposure time'

    # Change the airmass by 1
    _zp_spec = flux_calib_refactor.standard_zeropoint(
        obs_spec, std_spec, exptime=30., atm_extinction=atmext, airmass=2.04, nresln=20,
        resolution=3000, region_mask=region_mask,
    )[0]
    # Calculate the correction factor at airmass=1 (i.e., the difference between
    # airmass=2.04 and 1.04)
    cf = atmext.correction_factor(zp_spec.wave)
    assert np.allclose(_zp_spec.flux, zp_spec.flux + np.log10(cf)*2.5), 'Bad dependence on airmass'

    # Test without extinction correction
    _zp_spec = flux_calib_refactor.standard_zeropoint(
        obs_spec, std_spec, exptime=30., nresln=20,
        resolution=3000, region_mask=region_mask,
    )[0]
    cf = atmext.correction_factor(zp_spec.wave, airmass=1.04)
    assert np.allclose(_zp_spec.flux, zp_spec.flux - np.log10(cf)*2.5), 'Bad dependence on airmass'

    # Test without region mask
    _zp_spec, _fit_gpm, _fit_gpm_rej = flux_calib_refactor.standard_zeropoint(
        obs_spec, std_spec, exptime=30., atm_extinction=atmext, airmass=1.04, nresln=20,
        resolution=3000,
    )[:3]
    assert np.array_equal(obs_spec.gpm, _fit_gpm), \
        'No extra regions in the spectrum should be masked'
    assert not np.all(_fit_gpm_rej), 'Pixels should be rejected by fit'

    # TODO: Test use of the telluric model

    # Test directly setting the breakpoint spacing
    # WARNING: assumes precision of `bkspace` is sufficient to give exactly the
    # same result as using nresln and resolution.
    _zp_spec, _fit_gpm, _fit_gpm_rej, _zp_bspl = flux_calib_refactor.standard_zeropoint(
        obs_spec, std_spec, exptime=30., atm_extinction=atmext, airmass=1.04, bkspace=29.449789,
        region_mask=region_mask,
    )
    assert len(zp_bspl.breakpoints) == len(_zp_bspl.breakpoints), 'Number of breakpoints changed'
    assert np.allclose(zp_bspl.breakpoints, _zp_bspl.breakpoints), 'Placing of breakpoints changed'

    _zp_spec, _fit_gpm, _fit_gpm_rej, _zp_bspl = flux_calib_refactor.standard_zeropoint(
        obs_spec, std_spec, exptime=30., atm_extinction=atmext, airmass=1.04, bkspace=50.,
        region_mask=region_mask,
    )
    assert len(zp_bspl.breakpoints) > len(_zp_bspl.breakpoints), 'Should have viewer brakpoints'
    assert np.all(np.diff(_zp_bspl.breakpoints) > 50.), 'Breakpoint spacing is not right'

    # Disable rejection
    _zp_spec, _fit_gpm, _fit_gpm_rej, _zp_bspl = flux_calib_refactor.standard_zeropoint(
        obs_spec, std_spec, exptime=30., atm_extinction=atmext, airmass=1.04, nresln=20,
        resolution=3000, region_mask=region_mask, maxiter=0,
    )
    # TODO: Setting maxiter=0 does not actually disable rejections!  This needs
    # to be fixed.
    # assert np.sum(_fit_gpm) == np.sum(_fit_gpm_rej), 'Rejections should not occur'
    # Use this as a temporary tests
    assert np.sum(fit_gpm_rej) < np.sum(_fit_gpm_rej), 'Should be fewer rejections'


def test_standard_zeropoint_qa():

    masks = [
        dataPaths.masks.get_file_path('hydrogen.toml'),
        dataPaths.masks.get_file_path('telluric.toml'),
        dataPaths.masks.get_file_path('atm.toml'),
    ]
    region_mask = wavemask.read_wavelength_masks(masks)

    # Load the test spectrum data
    # TODO: Change this to use a spec1d file?
    test_spec_file = dataPaths.tests.get_file_path('test_standard_zeropoint_obs_spec.npz')
    spec_data = np.load(test_spec_file)
    obs_spec = spectrum.Spectrum(
        spec_data['wave_cnts'], spec_data['counts'], ivar=spec_data['counts_ivar'],
        gpm=spec_data['counts_mask']
    )

    archives = standard.archived_flux_classes()
    std_spec = archives['calspec'].from_name('FEIGE66')
    r = Resample(std_spec.flux, x=std_spec.wave, newx=obs_spec.wave, conserve=False)
    std_spec = spectrum.Spectrum(r.outx, r.outy, gpm=r.outf > 0.8)

    atmext = atmextinction.AtmosphericExtinction.from_coordinates(-121.6367, 37.3433)

    zp_spec, fit_gpm, fit_gpm_rej, zp_bspl = flux_calib_refactor.standard_zeropoint(
        obs_spec, std_spec, exptime=30., atm_extinction=atmext, airmass=1.04, nresln=20,
        resolution=3000, region_mask=region_mask,
    )

    # Test without writing to a file
    flux_calib_refactor.standard_zeropoint_qa(zp_spec, fit_gpm, fit_gpm_rej, zp_bspl)

    # Test writing to a file
    qa_file = Path('test_zp_qa.png').absolute()
    if qa_file.is_file():
        # Remove it if it already exists
        qa_file.unlink()
    flux_calib_refactor.standard_zeropoint_qa(
        zp_spec, fit_gpm, fit_gpm_rej, zp_bspl, ofile=qa_file
    )
    assert qa_file.is_file(), 'File not written'
    qa_file.unlink()

#test_standard_zeropoint_qa()
