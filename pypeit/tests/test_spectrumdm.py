from pathlib import Path

from astropy.io import fits
from IPython import embed
import numpy as np

from pypeit import spectrumdm
from pypeit.core import spectrum


def test_basic_init():
    wave = np.linspace(3000,6000,num=1000)
    flux = np.ones(wave.size, dtype=float)
    ivar = np.ones(wave.size, dtype=float)
    gpm = np.ones(wave.size, dtype=bool)
    meta = {'int': 1, 'flt': 3.4, 'str': 'test', 'bool': True}

    spec = spectrum.Spectrum(wave, flux, ivar=ivar, gpm=gpm, meta=meta)
    specdm = spectrumdm.SpectrumContainer(wave, flux, ivar=ivar, gpm=gpm, meta=meta)
    assert np.array_equal(spec.wave, specdm.wave), 'Wavelengths should be identical'

    # Include a name
    specdm = spectrumdm.SpectrumContainer(wave, flux, ivar=ivar, gpm=gpm, meta=meta, name='test')
    assert specdm.name == 'test', 'Name not set correctly'


def test_copy():
    wave = np.linspace(3000,6000,num=1000)
    flux = np.ones(wave.size, dtype=float)
    ivar = np.ones(wave.size, dtype=float)
    gpm = np.ones(wave.size, dtype=bool)
    meta = {'int': 1, 'flt': 3.4, 'str': 'test', 'bool': True}

    spec = spectrumdm.SpectrumContainer(wave, flux, ivar=ivar, gpm=gpm, meta=meta, name='test')
    _spec = spec.copy()
    assert isinstance(_spec, spectrumdm.SpectrumContainer), 'Copy has the wrong type'
    assert spec.name == _spec.name, 'Name is incorrect'
    assert np.array_equal(spec.wave, _spec.wave), 'Wavelengths should be identical'
    assert spec.meta['int'] == _spec.meta['int'], 'metadata changed'


def test_io():
    wave = np.linspace(3000,6000,num=1000)
    flux = np.ones(wave.size, dtype=float)
    ivar = np.ones(wave.size, dtype=float)
    gpm = np.ones(wave.size, dtype=bool)
    meta = {'int': 1, 'flt': 3.4, 'str': 'test', 'bool': True}
    name = 'test'

    spec = spectrumdm.SpectrumContainer(wave, flux, ivar=ivar, gpm=gpm, meta=meta, name=name)

    # Test writing to an HDUList
    hdu = spec.to_hdu(add_primary=True)
    assert len(hdu) == 2, 'HDUList should include header'
    assert isinstance(hdu[1], fits.BinTableHDU), 'Spectral data should be written to BinTableHDU'
    assert hdu[1].name == spec.name.upper(), 'Extension name is wrong'
    assert sorted(hdu[1].data.columns.names) == ['flux', 'gpm', 'ivar', 'wave'], 'Wrong columns'
    assert np.array_equal(hdu[1].data['wave'], wave), 'Wavelengths should be identical'
    assert 'METAKEYS' in hdu[1].header, 'metadata keys should be in header'
    assert (
        sorted(meta.keys())
        == sorted(map(lambda x : x.strip(), hdu[1].header['METAKEYS'].split(',')))
    ), 'Should be able to write all keys to the header'

    # Test writing to a file
    ofile = Path('test_spec.fits').absolute()
    if ofile.is_file():
        ofile.unlink()
    spec.to_file(ofile)
    assert ofile.is_file(), 'File not written'

    with fits.open(ofile) as _hdu:
        _spec = spectrumdm.SpectrumContainer.from_hdu(_hdu[1])
    assert _spec.name == name, 'Name should be the upper-case version of the original name'
    assert _spec.meta.keys() == spec.meta.keys(), 'Metadata keys should be the same'
    assert _spec.meta['int'] == spec.meta['int'], 'Metadata value changed'
    assert np.array_equal(_spec.wave, spec.wave), 'Wavelength arrays should be identical'

    # Test loading from a file
    _spec = spectrumdm.SpectrumContainer.from_file(ofile)
    assert _spec.name == name, 'Name should be the upper-case version of the original name'
    assert _spec.meta.keys() == spec.meta.keys(), 'Metadata keys should be the same'
    assert _spec.meta['int'] == spec.meta['int'], 'Metadata value changed'
    assert np.array_equal(_spec.wave, spec.wave), 'Wavelength arrays should be identical'

    ofile.unlink()


def test_to_from_spectrum():
    wave = np.linspace(3000,6000,num=1000)
    flux = np.ones(wave.size, dtype=float)
    ivar = np.ones(wave.size, dtype=float)
    gpm = np.ones(wave.size, dtype=bool)
    meta = {'int': 1, 'flt': 3.4, 'str': 'test', 'bool': True}

    spec = spectrum.Spectrum(wave, flux, ivar=ivar, gpm=gpm, meta=meta)
    specdm = spectrumdm.SpectrumContainer.from_spectrum(spec, name='test')

    assert np.array_equal(spec.wave, specdm.wave), 'Wavelengths should be identical'
    assert spec.meta.keys() == specdm.meta.keys(), 'Meta keys should be the same'

    _spec = specdm.to_spectrum()
    assert np.array_equal(_spec.wave, spec.wave), 'Wavelengths should be identical'
    assert _spec.meta.keys() == spec.meta.keys(), 'Meta keys should be the same'


# TODO: Need to test spectra with 2D flux arrays