
from IPython import embed
import numpy as np
import pytest

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

    zp_spec, fit_gpm, fit_gpm_rej, zp_model, zp_model_gpm = flux_calib_refactor.standard_zeropoint(
        obs_spec, std_spec, exptime=30., atm_extinction=atmext, airmass=1.04, nresln=20,
        resolution=3000, region_mask=region_mask, debug=True,
    )

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

    zp_spec, fit_gpm, fit_gpm_rej, zp_model, zp_model_gpm = flux_calib_refactor.standard_zeropoint(
        obs_spec, std_spec, exptime=30., atm_extinction=atmext, airmass=1.04, nresln=20,
        resolution=3000, region_mask=region_mask, debug=True,
    )

    # Change the exposure time
    _zp_spec = flux_calib_refactor.standard_zeropoint(
        obs_spec, std_spec, exptime=10., atm_extinction=atmext, airmass=1.04, nresln=20,
        resolution=3000, region_mask=region_mask, debug=True,
    )[0]
    assert np.allclose(zp_spec.flux - _zp_spec.flux + 2.5*np.log10(3.), 0.)

    # Change the airmass
    _zp_spec = flux_calib_refactor.standard_zeropoint(
        obs_spec, std_spec, exptime=10., atm_extinction=atmext, airmass=2.08, nresln=20,
        resolution=3000, region_mask=region_mask, debug=True,
    )[0]
    embed()
    exit()
    

test_standard_zeropoint_kwargs()
