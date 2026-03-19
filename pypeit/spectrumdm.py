"""
Implements a spectrum data container
"""

from astropy.io import fits
from IPython import embed
import numpy as np

from pypeit import datamodel
from pypeit import log
from pypeit.core import spectrum


class SpectrumContainer(datamodel.DataContainer, spectrum.Spectrum):

    version = '1.0.0'

    datamodel = {
        'wave' : dict(otype=np.ndarray, atype=np.floating, descr='Wavelength array'),
        'flux' : dict(otype=np.ndarray, atype=np.floating, descr='Flux array'),
        'ivar' : dict(otype=np.ndarray, atype=np.floating, descr='Inverse variance in flux'),
        'gpm' : dict(otype=np.ndarray, atype=(bool, np.bool), descr='Good-pixel mask'),
    }

    internals = [
        'name', 'meta'
    ]

    allowed_metadata_types = (int, np.integer, float, np.floating, bool, np.bool, str)

    def __init__(self, *args, name=None, **kwargs):
        datamodel.DataContainer.__init__(self)
        self.name = name
        spectrum.Spectrum.__init__(self, *args, **kwargs)

    def copy(self):
        """
        Override the Spectrum copy method.
        """
        # NOTE: This should call the Spectrum.copy method (i.e., there is no
        # DataContainer.copy method).
        cp = super().copy()
        cp.name = self.name
        return cp

    def _bundle(self, ext=None, transpose_arrays=False):
        """
        Override the DataContainer method.
        """
        # Set the name of the HDU to self.name if `ext` is not provided.
        if ext is None:
            ext = self.name
        # If it is still none, set it to 'SPECTRUM'
        if ext is None:
            ext = 'SPECTRUM'

        # Use the base class function, which will just collect the datamodel
        # arrays into a dictionary.  The returned object is a list with one
        # entry, a dictionary with one key (ext), that itself contains a
        # dictionary with the spectrum vectors.
        data = super()._bundle(ext=ext, transpose_arrays=transpose_arrays)

        # The extension name and the spectrum name should be the same, but save
        # the original
        if self.name is not None:
            data[0][ext]['NAME'] = self.name

        # Add the metadata
        if self.meta is not None:
            meta_keys = []
            for key, value in self.meta.items():
                if not isinstance(self.meta[key], self.allowed_metadata_types):
                    log.warning(
                        f'{key} metadata is not a type ({type(self.meta[key])}) that can be '
                        'parsed into the header of a FITS extension.'
                    )
                    continue
                data[0][ext][key] = value
                meta_keys += [key]
            # Finally add the list of metadata keys that were included
            data[0][ext]['METAKEYS'] = ', '.join(meta_keys)

        return data

    @classmethod
    def from_hdu(cls, hdu, chk_version=True, **kwargs):
        """
        Override base class function to parse data from the header into the
        metadata dictionary.
        """
        # This reproduces *all* of the lines in the base class function.  We
        # need to know which hdus were parsed to setup the metadata dictionary
        # and name.

        # The following three lines are identical to DataContainer.from_hdu
        d, dm_version_passed, dm_type_passed, parsed_hdus = cls._parse(hdu, **kwargs)
        cls._check_parsed(dm_version_passed, dm_type_passed, chk_version=chk_version)
        self = cls.from_dict(d=d)

        # Only one hdu should be parsed
        if len(parsed_hdus) != 1:
            log.warning(
                f'The {cls.__name__} data should be contained in a single HDU, but multiple HDUs '
                'were parsed.  Just taking data from the first one.'
            )
        parsed_hdus = parsed_hdus[0]

        # Get the header
        hdr = hdu[parsed_hdus].header if isinstance(hdu, fits.HDUList) else hdu.header

        if 'NAME' in hdr:
            # If the name was in the header, use it
            self.name = hdr['NAME']
        elif isinstance(parsed_hdus, str):
            # Or, if the extension identifier is a string (instead of an
            # integer), use that
            self.name = parsed_hdus

        # Parse the metadata, if available
        if 'METAKEYS' in hdr:
            keys = list(map(lambda x : x.strip(), hdr['METAKEYS'].split(',')))
            self.meta = {key : hdr[key.upper()] for key in keys}

        return self

    @classmethod
    def from_spectrum(cls, spec, name=None):
        return cls(spec.wave, spec.flux, ivar=spec.ivar, gpm=spec.gpm, meta=spec.meta, name=name)

    def to_spectrum(self):
        return spectrum.Spectrum(
            self.wave, self.flux, ivar=self.ivar, gpm=self.gpm, meta=self.meta
        )
